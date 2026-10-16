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
