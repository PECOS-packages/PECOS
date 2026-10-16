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

use pecos_neo::extensible::DecompositionNoiseStrategy;
use pecos_neo::noise::CorrelatedNoiseChannel;
use pecos_neo::noise::composite::prelude::*;
use pecos_neo::noise::composite::{BatchCompositeChannel, CompositeNoiseModelBuilder, Prob};
use pecos_neo::noise::validation::ValidationError;

fn rejects(error: Option<ValidationError>, parameter: &str) {
    let error = error.expect("invalid input must be rejected");
    assert!(error.to_string().contains(parameter), "{error}");
}

#[test]
fn composite_probability_rejects_invalid_input() {
    for p in [f64::NAN, -0.1, 1.1] {
        rejects(CompositeNoiseModelBuilder::new().with_p1(p).err(), "p1");
    }
}

#[test]
fn leakage_scale_rejects_invalid_input() {
    for p in [f64::NAN, -0.1, 1.1] {
        rejects(
            CompositeNoiseModelBuilder::new()
                .with_leakage_scale(p)
                .err(),
            "leakage_scale",
        );
    }
}

#[test]
fn primitive_probability_rejects_invalid_input() {
    for p in [f64::NAN, -0.1, 1.1] {
        rejects(Prob::new(p, nothing()).err(), "probability");
    }
}

#[test]
fn correlated_probability_rejects_invalid_input() {
    for p in [f64::NAN, -0.1, 1.1] {
        rejects(CorrelatedNoiseChannel::new(0.1, p).err(), "correlation");
        rejects(
            CorrelatedNoiseChannel::new(p, 0.1).err(),
            "base_error_probability",
        );
    }
}

#[test]
fn blended_weight_rejects_invalid_input() {
    for p in [f64::NAN, -0.1, 1.1] {
        rejects(
            DecompositionNoiseStrategy::blended(p).err(),
            "high_level_weight",
        );
    }
}

#[test]
fn emission_probability_rejects_invalid_input() {
    for p in [f64::NAN, -0.1, 1.1] {
        rejects(sample_emission_with_prob(p).err(), "prob");
        rejects(
            pecos_neo::noise::composite::IndependentEmissionWithPartnerDepolarize::new(p).err(),
            "emission_prob",
        );
        rejects(amplitude_damping_biased(p).err(), "x_probability");
    }
}

#[test]
fn batch_probability_rejects_invalid_input() {
    for p in [f64::NAN, -0.1, 1.1] {
        rejects(
            BatchCompositeChannel::new("test", p, nothing()).err(),
            "probability",
        );
    }
}

#[test]
fn traditional_channels_reject_invalid_probabilities() {
    use pecos_neo::noise::*;
    for p in [f64::NAN, -0.1, 1.1, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(
            SingleQubitChannel::new(
                p,
                PauliWeights::uniform(),
                0.0,
                SingleQubitEmissionWeights::uniform(),
                0.0
            )
            .is_err()
        );
        assert!(SingleQubitChannel::depolarizing(p).is_err());
        assert!(SingleQubitChannel::dephasing(p).is_err());
        assert!(SingleQubitChannel::bit_flip(p).is_err());
        assert!(MeasurementChannel::symmetric(p).is_err());
        assert!(MeasurementChannel::asymmetric(p, 0.0).is_err());
        assert!(MeasurementChannel::asymmetric(0.0, p).is_err());
        assert!(MeasurementStateFlipChannel::new(p).is_err());
        assert!(PreparationChannel::new(p).is_err());
        assert!(PreparationChannel::default().with_leakage(p).is_err());
        assert!(CrosstalkChannel::new().with_local_rate(p).is_err());
        assert!(CrosstalkChannel::new().with_global_rate(p).is_err());
        assert!(CrosstalkChannel::new().with_leakage(p).is_err());
        assert!(CrosstalkChannel::global_only(p).is_err());
        assert!(
            TwoQubitChannel::new(
                p,
                AngleScaling::constant(),
                TwoQubitPauliWeights::uniform(),
                0.0,
                TwoQubitEmissionWeights::uniform_pauli(),
                0.0
            )
            .is_err()
        );
        assert!(TwoQubitChannel::depolarizing(p).is_err());
        assert!(SingleQubitChannel::default().with_emission(p, 0.0).is_err());
        assert!(SingleQubitChannel::default().with_emission(0.0, p).is_err());
        assert!(
            SingleQubitChannel::default()
                .with_emission_weights(p, SingleQubitEmissionWeights::uniform())
                .is_err()
        );
        assert!(SingleQubitChannel::default().with_seepage(p).is_err());
        assert!(
            TwoQubitChannel::default()
                .with_emission_weights(p, TwoQubitEmissionWeights::uniform_pauli())
                .is_err()
        );
        assert!(TwoQubitChannel::default().with_leakage(p).is_err());
        assert!(TwoQubitChannel::default().with_seepage(p).is_err());
    }
}

#[test]
fn pattern_probabilities_are_validated_even_when_disabled() {
    use pecos_neo::noise::patterns::{DeviceNoiseParams, realistic_device_noise, with_leakage};
    for p in [f64::NAN, -0.1, 1.1] {
        for args in [
            (p, 0.0, 0.0, 0.0),
            (0.0, p, 0.0, 0.0),
            (0.0, 0.0, p, 0.0),
            (0.0, 0.0, 0.0, p),
        ] {
            assert!(with_leakage(args.0, args.1, args.2, args.3).is_err());
        }
        assert!(DeviceNoiseParams::new().with_leakage(p).is_err());
        assert!(DeviceNoiseParams::new().with_prep_error(p).is_err());
        assert!(DeviceNoiseParams::new().with_seepage(p).is_err());
        for field in 0..7 {
            let mut params = DeviceNoiseParams::new();
            match field {
                0 => params.p1 = p,
                1 => params.p2 = p,
                2 => params.p_meas_01 = p,
                3 => params.p_meas_10 = p,
                4 => params.p_prep = p,
                5 => params.leakage_rate = p,
                _ => params.seepage_rate = p,
            }
            assert!(realistic_device_noise(&params).is_err());
        }
    }
}

#[test]
fn sample_weights_require_a_finite_positive_total() {
    use pecos_neo::noise::composite::{BoxSample, Sample};
    for weights in [
        vec![],
        vec![0.0],
        vec![f64::NAN],
        vec![-0.1, 1.1],
        vec![f64::INFINITY],
        vec![f64::MAX, f64::MAX],
    ] {
        assert!(Sample::new(weights.iter().map(|w| (*w, nothing())).collect()).is_err());
        assert!(
            BoxSample::new(
                weights
                    .iter()
                    .map(|w| (*w, Box::new(nothing()) as Box<dyn Primitive>))
                    .collect()
            )
            .is_err()
        );
    }
    assert!(Sample::new(vec![(2.0, nothing()), (3.0, nothing())]).is_ok());
}

#[test]
fn probability_threshold_and_biased_outcomes_reject_invalid_inputs() {
    use pecos_neo::noise::composite::{RandomOutcome, batch::ProbabilityThreshold};
    let rng = pecos_random::PecosRng::seed_from_u64(42);
    for p in [f64::NAN, -0.1, 1.1] {
        assert!(ProbabilityThreshold::new(p, &rng).is_err());
        assert!(RandomOutcome::new(p).is_err());
        assert!(RandomOutcome::biased_one(p).is_err());
        assert!(random_outcome_biased(p).is_err());
    }
}

#[test]
fn distributions_reject_invalid_weights_and_ratios() {
    use pecos_neo::noise::{
        CrosstalkTransitions, PauliWeights as TraditionalPauliWeights, SingleQubitEmissionWeights,
        TwoQubitEmissionWeights, TwoQubitPauliWeights,
    };
    for p in [f64::NAN, -0.1, 1.1] {
        assert!(TraditionalPauliWeights::z_biased(p).is_err());
        assert!(TraditionalPauliWeights::x_biased(p).is_err());
        assert!(TwoQubitPauliWeights::zz_biased(p).is_err());
        assert!(TraditionalPauliWeights::custom(p, 1.0 - p, 0.0).is_err());
        assert!(SingleQubitEmissionWeights::custom(p, 1.0 - p, 0.0, 0.0).is_err());
        let mut two_pauli = [0.0; 15];
        two_pauli[0] = p;
        two_pauli[1] = 1.0 - p;
        assert!(TwoQubitPauliWeights::custom(two_pauli).is_err());
        let mut two_emission = [0.0; 24];
        two_emission[0] = p;
        two_emission[1] = 1.0 - p;
        assert!(TwoQubitEmissionWeights::custom(two_emission).is_err());
        assert!(CrosstalkTransitions::custom(p, 1.0 - p, 0.0, 1.0, 0.0, 0.0).is_err());
        assert!(emission_with_leakage(p).is_err());
    }
    // Relative composite weights may exceed one; only their domain and total are constrained.
    assert!(PauliWeights::custom(2.0, 3.0, 0.0).is_ok());
    for weights in [
        [f64::NAN, 1.0, 0.0],
        [-0.1, 1.1, 0.0],
        [f64::INFINITY, 0.0, 0.0],
        [0.0; 3],
        [f64::MAX; 3],
    ] {
        assert!(PauliWeights::custom(weights[0], weights[1], weights[2]).is_err());
    }
}

#[test]
fn scale_setters_and_plugins_validate_before_sampling() {
    use pecos_neo::noise::plugins::{DepolarizingPlugin, MeasurementNoisePlugin};
    use pecos_neo::noise::*;
    for p in [f64::NAN, -0.1, 1.1] {
        assert!(
            SingleQubitChannel::depolarizing(1.0)
                .unwrap()
                .with_scale(p)
                .is_err()
        );
        assert!(
            TwoQubitChannel::depolarizing(1.0)
                .unwrap()
                .with_scale(p)
                .is_err()
        );
        assert!(
            MeasurementChannel::symmetric(1.0)
                .unwrap()
                .with_scale(p)
                .is_err()
        );
        assert!(LeakageChannel::new().with_scale(p).is_err());
        assert!(DepolarizingPlugin::new(p, 0.0).is_err());
        assert!(DepolarizingPlugin::new(0.0, p).is_err());
        assert!(MeasurementNoisePlugin::symmetric(p).is_err());
        assert!(MeasurementNoisePlugin::asymmetric(0.0, p).is_err());
        assert!(NoiseModelBuilder::new().with_p1(p).build().is_err());
        assert!(GeneralNoiseModelBuilder::new().with_p1(p).build().is_err());
    }
}

#[test]
fn category_and_gate_probabilities_reject_invalid_inputs() {
    use pecos_neo::command::GateType;
    use pecos_neo::extensible::{GateCategory, GateId};
    use pecos_neo::noise::{
        CategoryBasedChannel, GateDependentChannel, GateIdDependentChannel, GateIdNoiseConfig,
        GateNoiseConfig, PauliWeights,
    };
    for p in [f64::NAN, -0.1, 1.1, f64::INFINITY] {
        assert!(
            CategoryBasedChannel::new()
                .with_category(GateCategory::Idle, p)
                .is_err()
        );
        assert!(
            CategoryBasedChannel::new()
                .with_category_weights(GateCategory::Idle, p, PauliWeights::uniform())
                .is_err()
        );
        assert!(CategoryBasedChannel::new().with_default(p).is_err());
        assert!(GateNoiseConfig::new(p).is_err());
        assert!(
            GateDependentChannel::new()
                .with_gate_error(GateType::H, p)
                .is_err()
        );
        assert!(GateDependentChannel::new().with_default(p).is_err());
        assert!(GateIdNoiseConfig::new(p).is_err());
        assert!(
            GateIdDependentChannel::new()
                .with_gate_error(GateId(0), p)
                .is_err()
        );
        assert!(
            GateIdDependentChannel::new()
                .with_gate_type_error(GateType::H, p)
                .is_err()
        );
        assert!(GateIdDependentChannel::new().with_default(p).is_err());
    }
}

#[test]
fn extensible_probability_setters_reject_before_mutation() {
    use pecos_neo::extensible::{
        GateCategory, GateDefinitions, GateIdNoiseConfig, GateNoiseParams, gates,
    };
    for p in [f64::NAN, -0.1, 1.1] {
        assert!(GateNoiseParams::with_error(p).is_err());
        assert!(GateIdNoiseConfig::new().with_global_default(p).is_err());
        assert!(
            GateIdNoiseConfig::new()
                .with_category_default(GateCategory::Idle, p)
                .is_err()
        );
        assert!(GateIdNoiseConfig::new().with_typical_rates(p, 0.0).is_err());
        assert!(GateIdNoiseConfig::new().with_typical_rates(0.0, p).is_err());
        let mut config = GateIdNoiseConfig::new();
        config.set_gate_error(gates::H, 0.25).unwrap();
        assert!(config.set_gate_error(gates::H, p).is_err());
        assert!((config.get_error_probability(gates::H, None) - 0.25).abs() < f64::EPSILON);
        let mut defs = GateDefinitions::new();
        defs.set_noise_error(gates::H, 0.25).unwrap();
        assert!(defs.set_noise_error(gates::H, p).is_err());
        assert!((defs.error_probability(gates::H) - 0.25).abs() < f64::EPSILON);
    }
}

#[test]
fn importance_and_subset_probabilities_reject_invalid_inputs() {
    use pecos_neo::sampling::subset::{
        BernoulliSubsetSimulation, ProperSubsetSimulation, SubsetConfig,
    };
    use pecos_neo::sampling::{ImportanceConfig, ImportanceSamplingNoise};
    use pecos_neo::tool::importance_sampling;
    for p in [f64::NAN, -0.1, 1.1] {
        assert!(ImportanceConfig::new(p, 0.5).is_err());
        assert!(ImportanceConfig::new(0.5, p).is_err());
        assert!(ImportanceConfig::with_boost(p, 10.0).is_err());
        assert!(
            ImportanceSamplingNoise::new()
                .with_single_qubit(p, 1.0)
                .is_err()
        );
        assert!(
            ImportanceSamplingNoise::new()
                .with_two_qubit(p, 1.0)
                .is_err()
        );
        assert!(
            ImportanceSamplingNoise::new()
                .with_measurement(p, 1.0)
                .is_err()
        );
        assert!(importance_sampling(1).with_p1(p).is_err());
        assert!(importance_sampling(1).with_p2(p).is_err());
        assert!(importance_sampling(1).with_p_meas(p).is_err());
        assert!(importance_sampling(1).with_uniform_error(p).is_err());
        assert!(BernoulliSubsetSimulation::new(p, 10, 1.0).is_err());
        assert!(ProperSubsetSimulation::new(p, 1.0, 1.0, 10, SubsetConfig::default()).is_err());
    }
    for boost in [f64::NAN, -1.0, f64::INFINITY] {
        assert!(ImportanceConfig::with_boost(0.1, boost).is_err());
        assert!(importance_sampling(1).with_boost(boost).is_err());
    }
}

#[test]
fn per_gate_vectors_and_angle_extrema_are_validated() {
    use pecos_neo::command::GateType;
    use pecos_neo::noise::{AngleScaling, PerGatePauliChannel, TwoQubitChannel};
    for p in [f64::NAN, -0.1, 1.1] {
        assert!(PerGatePauliChannel::new().with_base(p, 0.0).is_err());
        assert!(PerGatePauliChannel::new().with_base(0.0, p).is_err());
        assert!(PerGatePauliChannel::new().with_meas_init(p, 0.0).is_err());
        assert!(PerGatePauliChannel::new().with_meas_init(0.0, p).is_err());
        assert!(
            PerGatePauliChannel::new()
                .with_1q_rates(GateType::H, [p, 0.0, 0.0])
                .is_err()
        );
    }
    assert!(
        PerGatePauliChannel::new()
            .with_1q_rates(GateType::H, [0.4; 3])
            .is_err()
    );
    // Endpoints are zero, but 8*t - 8*t^2 peaks at 2 in the interior.
    assert!(
        TwoQubitChannel::depolarizing(1.0)
            .unwrap()
            .with_angle_scaling(AngleScaling::polynomial(0.0, 8.0, -8.0, 2.0))
            .is_err()
    );
    // Signed coefficients are allowed when the entire curve stays in range.
    assert!(
        TwoQubitChannel::depolarizing(1.0)
            .unwrap()
            .with_angle_scaling(AngleScaling::polynomial(0.0, 4.0, -4.0, 2.0))
            .is_ok()
    );
    assert!(
        TwoQubitChannel::depolarizing(0.1)
            .unwrap()
            .with_angle_scaling(AngleScaling::polynomial(f64::NAN, 0.0, 0.0, 1.0))
            .is_err()
    );
}

#[test]
fn leakage_plugin_and_read_accessors_preserve_validation() {
    use pecos_neo::noise::plugins::LeakagePlugin;
    use pecos_neo::noise::{MeasurementChannel, SingleQubitChannel};
    for p in [f64::NAN, -0.1, 1.1] {
        assert!(LeakagePlugin::with_scale(p).is_err());
    }
    let channel = SingleQubitChannel::depolarizing(0.25).unwrap();
    assert!((channel.error_probability() - 0.25).abs() < f64::EPSILON);
    assert!(channel.clone().with_scale(5.0).is_err());
    assert!((channel.error_probability() - 0.25).abs() < f64::EPSILON);
    let measurement = MeasurementChannel::asymmetric(0.25, 0.5).unwrap();
    assert!((measurement.p_meas_0_to_1() - 0.25).abs() < f64::EPSILON);
    assert!((measurement.p_meas_1_to_0() - 0.5).abs() < f64::EPSILON);
}

#[test]
fn dynamic_probabilities_and_angle_helpers_validate_before_sampling() {
    for p in [f64::NAN, -0.1, 1.1] {
        assert!(Probability::new(p).is_err());
        assert!(prob_angle_scaled(p, 1.0, nothing()).is_err());
        assert!(prob_angle_linear(p, nothing()).is_err());
        assert!(prob_angle_quadratic(p, nothing()).is_err());
    }
    for power in [f64::NAN, -1.0, f64::INFINITY] {
        assert!(prob_angle_scaled(0.1, power, nothing()).is_err());
    }
    assert!(prob_angle_quadratic(0.5, nothing()).is_err());
    assert!(prob_angle_quadratic(0.25, nothing()).is_ok());
}

#[test]
fn resampling_rejects_invalid_weights_before_mutating_entities() {
    use pecos_neo::ecs::World;
    use pecos_neo::sampling::SampleWeight;
    use pecos_simulators::SparseStab;
    for log_weight in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut world: World<SparseStab> = World::new(42);
        let entity = world.spawn_with_simulator(SparseStab::with_seed(1, 42));
        world.weights.get_mut(entity).unwrap().weight = SampleWeight::from_log(log_weight);
        let mut rng = pecos_random::PecosRng::seed_from_u64(42);
        let error = world.resample_by_weight(2, &mut rng).unwrap_err();
        assert!(error.to_string().contains("entity sampling weights"));
        assert!(world.is_alive(entity));
        assert_eq!(world.active_entities(), vec![entity]);
    }
}
