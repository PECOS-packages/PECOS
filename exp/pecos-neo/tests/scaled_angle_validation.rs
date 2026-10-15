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

use pecos_neo::noise::composite::Nothing;
use pecos_neo::noise::composite::{CompositeNoiseModelBuilder, ProbQuadratic};
use pecos_neo::noise::validation::MAX_ANGLE_RATE;
use pecos_neo::noise::{GeneralNoiseModelBuilder, IdleChannel, NoiseModelBuilder};
use std::collections::BTreeMap;

#[test]
fn idle_quadratic_rejects_scaled_overflow_in_both_orders() {
    assert!(
        IdleChannel::default()
            .with_quadratic_rate(MAX_ANGLE_RATE)
            .unwrap()
            .with_coherent_to_incoherent_factor(2.0)
            .is_err()
    );
    assert!(
        IdleChannel::default()
            .with_coherent_to_incoherent_factor(2.0)
            .unwrap()
            .with_quadratic_rate(MAX_ANGLE_RATE)
            .is_err()
    );
}

#[test]
fn idle_sine_rejects_scaled_overflow_on_every_axis() {
    for axis in ["X", "Y", "Z", "L"] {
        assert!(
            IdleChannel::default()
                .with_sin_squared(MAX_ANGLE_RATE, BTreeMap::from([(axis.to_owned(), 2.0)]))
                .is_err()
        );
    }
}

#[test]
fn primitive_quadratic_rejects_scaled_overflow() {
    assert!(
        ProbQuadratic::new(MAX_ANGLE_RATE, Nothing)
            .unwrap()
            .with_factor(2.0)
            .is_err()
    );
    assert!(ProbQuadratic::new(MAX_ANGLE_RATE * 2.0, Nothing).is_err());
}

#[test]
fn composite_builder_rejects_scaled_overflow_in_both_orders() {
    assert!(
        CompositeNoiseModelBuilder::new()
            .with_p_idle_quadratic(MAX_ANGLE_RATE)
            .unwrap()
            .with_coherent_to_incoherent_factor(2.0)
            .is_err()
    );
    assert!(
        CompositeNoiseModelBuilder::new()
            .with_coherent_to_incoherent_factor(2.0)
            .unwrap()
            .with_p_idle_quadratic(MAX_ANGLE_RATE)
            .is_err()
    );
    assert!(
        CompositeNoiseModelBuilder::new()
            .with_coherent_to_incoherent_factor(2.0)
            .unwrap()
            .with_idle_noise(0.0, MAX_ANGLE_RATE)
            .is_err()
    );
}

#[test]
fn general_builder_rejects_scaled_overflow_in_both_orders() {
    assert!(
        GeneralNoiseModelBuilder::new()
            .with_p_idle_quadratic(MAX_ANGLE_RATE)
            .unwrap()
            .with_p_idle_coherent_to_incoherent_factor(2.0)
            .is_err()
    );
    assert!(
        GeneralNoiseModelBuilder::new()
            .with_p_idle_coherent_to_incoherent_factor(2.0)
            .unwrap()
            .with_p_idle_quadratic(MAX_ANGLE_RATE)
            .is_err()
    );
}

#[test]
fn general_builder_rejects_sine_scaled_overflow() {
    for axis in ["X", "Y", "Z", "L"] {
        assert!(
            GeneralNoiseModelBuilder::new()
                .with_p_idle_sin_squared(MAX_ANGLE_RATE, &BTreeMap::from([(axis.to_owned(), 2.0)]))
                .is_err()
        );
    }
}

#[test]
fn noise_builder_rejects_scaled_overflow_in_both_orders() {
    assert!(
        NoiseModelBuilder::new()
            .with_idle_noise(0.0, MAX_ANGLE_RATE)
            .unwrap()
            .with_coherent_idle(2.0)
            .is_err()
    );
    assert!(
        NoiseModelBuilder::new()
            .with_coherent_idle(2.0)
            .unwrap()
            .with_idle_noise(0.0, MAX_ANGLE_RATE)
            .is_err()
    );
}

#[test]
fn scaled_overflow_errors_name_both_parameters() {
    let error = IdleChannel::default()
        .with_quadratic_rate(MAX_ANGLE_RATE)
        .unwrap()
        .with_coherent_to_incoherent_factor(f64::MAX)
        .err()
        .unwrap();
    assert!(error.field.contains("quadratic_rate"));
    assert!(error.field.contains("coherent_to_incoherent_factor"));
    let error = IdleChannel::default()
        .with_sin_squared(MAX_ANGLE_RATE, BTreeMap::from([("L".to_owned(), 2.0)]))
        .err()
        .unwrap();
    assert!(error.field.contains("sin_squared_rate"));
    assert!(error.field.contains("multiplier for 'L'"));
    let error = ProbQuadratic::new(MAX_ANGLE_RATE, Nothing)
        .unwrap()
        .with_factor(2.0)
        .err()
        .unwrap();
    assert!(error.field.contains("rate_per_time_unit"));
    assert!(error.field.contains("factor"));
}
