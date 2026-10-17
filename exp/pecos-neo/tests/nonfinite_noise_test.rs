// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0

use pecos_neo::command::GateType;
use pecos_neo::noise::composite::prelude::*;
use pecos_neo::noise::composite::{CoherentRotation, InjectCoherentRZ, OverRotation, ZZDephasing};
use pecos_neo::noise::{
    ComposableNoiseModel, GeneralNoiseModelBuilder, IdleChannel, NoiseModelBuilder,
};
use std::collections::BTreeMap;

#[test]
fn noise_construction_rejects_nonfinite_parameters() {
    let linear = BTreeMap::from([("Z".to_owned(), 1.0)]);
    macro_rules! check {
        ($value:ident, $expression:expr) => {
            for $value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.25] {
                let result = $expression;
                if $value.is_finite() {
                    assert!(result.is_ok(), "{}", stringify!($expression));
                } else {
                    let error = result.err().expect(stringify!($expression)).to_string();
                    assert!(error.contains(&$value.to_string()), "{error}");
                }
            }
        };
    }
    check!(v, InjectCoherentRZ::new(v));
    check!(v, CoherentRotation::new(GateType::RX, v));
    check!(v, CoherentRotation::rx(v));
    check!(v, CoherentRotation::ry(v));
    check!(v, CoherentRotation::rz(v));
    check!(v, OverRotation::new(GateType::RZ, v));
    check!(v, OverRotation::rz(v));
    check!(v, ZZDephasing::new(v));
    check!(v, ZZDephasing::from_rate(v));
    check!(v, coherent_rz(v));
    check!(v, coherent_rx(v));
    check!(v, coherent_ry(v));
    check!(v, coherent_rz_fixed(v));
    check!(v, over_rotation(GateType::RZ, v));
    check!(v, over_rotation_rz(v));
    check!(v, zz_dephasing(v));
    check!(v, zz_dephasing_rate(v));
    check!(v, prob_linear(v, inject_z()));
    check!(v, prob_quadratic(v, inject_z()));
    check!(v, prob_quadratic(0.1, inject_z()).unwrap().with_factor(v));
    check!(v, IdleChannel::default().with_quadratic_rate(v));
    check!(
        v,
        IdleChannel::default().with_sin_squared(v, linear.clone())
    );
    check!(
        v,
        IdleChannel::default().with_sin_squared(0.1, BTreeMap::from([("Z".to_owned(), v)]))
    );
    check!(
        v,
        GeneralNoiseModelBuilder::new()
            .with_p_idle_sin_squared(0.1, &BTreeMap::from([("Z".to_owned(), v)]))
    );
    check!(v, CompositeNoiseModelBuilder::new().with_idle_noise(0.1, v));
    check!(
        v,
        pecos_neo::noise::patterns::DeviceNoiseParams::new().with_t1(v)
    );
    check!(
        v,
        pecos_neo::noise::patterns::DeviceNoiseParams::new().with_t2(v)
    );
    check!(
        v,
        pecos_neo::noise::patterns::realistic_device_noise(
            &pecos_neo::noise::patterns::DeviceNoiseParams {
                t1_rate: v,
                ..Default::default()
            }
        )
    );
    check!(
        v,
        pecos_neo::noise::patterns::realistic_device_noise(
            &pecos_neo::noise::patterns::DeviceNoiseParams {
                t2_rate: v,
                ..Default::default()
            }
        )
    );
    check!(
        v,
        GeneralNoiseModelBuilder::new()
            .with_time_scale(pecos_core::TimeScale::NANOSECONDS)
            .with_idle_t1_t2(v, 1e-6)
    );
    check!(
        v,
        GeneralNoiseModelBuilder::new()
            .with_time_scale(pecos_core::TimeScale::NANOSECONDS)
            .with_idle_t1_t2(1.0, v)
    );
    check!(
        v,
        CompositeNoiseModelBuilder::new()
            .with_time_scale(pecos_core::TimeScale::NANOSECONDS)
            .with_idle_t1_t2(v, 1e-6)
    );
    check!(
        v,
        CompositeNoiseModelBuilder::new()
            .with_time_scale(pecos_core::TimeScale::NANOSECONDS)
            .with_idle_t1_t2(1.0, v)
    );
    check!(
        v,
        ComposableNoiseModel::new()
            .with_time_scale(pecos_core::TimeScale::NANOSECONDS)
            .with_idle_t1_t2(1.0, v)
    );
    check!(v, IdleChannel::linear(v));
    check!(
        v,
        IdleChannel::default().with_coherent_to_incoherent_factor(v)
    );
    check!(v, IdleChannel::default().with_idle_after_2q(v));
    check!(v, IdleChannel::from_t1_t2(v, 0.25));
    check!(v, IdleChannel::from_t1_t2(1.0, v));
    check!(v, NoiseModelBuilder::new().with_idle_noise(v, 0.1));
    check!(v, NoiseModelBuilder::new().with_idle_noise(0.1, v));
    check!(v, NoiseModelBuilder::new().with_coherent_idle(v));
    check!(v, NoiseModelBuilder::new().with_idle_after_2q(v));
    check!(
        v,
        GeneralNoiseModelBuilder::new().with_p_idle_linear(v, &linear)
    );
    check!(v, GeneralNoiseModelBuilder::new().with_p_idle_quadratic(v));
    check!(
        v,
        GeneralNoiseModelBuilder::new().with_p_idle_sin_squared(v, &linear)
    );
    check!(
        v,
        GeneralNoiseModelBuilder::new().with_p_idle_coherent_to_incoherent_factor(v)
    );
    check!(v, GeneralNoiseModelBuilder::new().with_idle_after_2q(v));
    check!(v, CompositeNoiseModelBuilder::new().with_p_idle_linear(v));
    check!(
        v,
        CompositeNoiseModelBuilder::new().with_p_idle_quadratic(v)
    );
    check!(
        v,
        CompositeNoiseModelBuilder::new().with_coherent_to_incoherent_factor(v)
    );
    check!(v, CompositeNoiseModelBuilder::new().with_idle_noise(v, 0.1));
    check!(v, CompositeNoiseModelBuilder::new().with_idle_after_2q(v));
    check!(
        v,
        ComposableNoiseModel::new()
            .with_time_scale(pecos_core::TimeScale::NANOSECONDS)
            .with_idle_t1_t2(v, 1e-6)
    );
}

#[test]
fn runtime_angle_products_reject_overflow_at_construction() {
    let duration_bound = pecos_core::TimeUnits::new(u64::MAX).as_f64();
    let rate = (f64::MAX / duration_bound).next_up();
    let fraction = (f64::MAX / std::f64::consts::TAU).next_up();
    let rejected = [
        InjectCoherentRZ::new(rate).is_err(),
        ZZDephasing::from_rate(rate).is_err(),
        IdleChannel::default().with_quadratic_rate(rate).is_err(),
        OverRotation::rz(fraction).is_err(),
    ];
    assert_eq!(rejected, [true; 4]);
}

#[test]
fn runtime_angle_products_at_bound_accept_maximum_duration() {
    use pecos_core::{Angle64, QubitId, TimeUnits};
    use pecos_neo::noise::{NoiseChannel, NoiseContext, NoiseEvent};
    use pecos_random::PecosRng;

    let duration = TimeUnits::new(u64::MAX);
    let rate = f64::MAX / duration.as_f64();
    let fraction = f64::MAX / std::f64::consts::TAU;
    let mut ctx = NoiseContext::new();
    ctx.set_current_idle(duration);
    ctx.set_current_gate(GateType::RZ, &[Angle64::new(u64::MAX)], 1);
    let mut rng = PecosRng::seed_from_u64(42);
    GateAction::apply(
        &InjectCoherentRZ::new(rate).unwrap(),
        QubitId(0),
        &mut ctx,
        &mut rng,
    );
    GateAction::apply(
        &ZZDephasing::from_rate(rate).unwrap(),
        QubitId(0),
        &mut ctx,
        &mut rng,
    );
    GateAction::apply(
        &OverRotation::rz(fraction).unwrap(),
        QubitId(0),
        &mut ctx,
        &mut rng,
    );
    IdleChannel::default()
        .with_quadratic_rate(rate)
        .unwrap()
        .with_coherent_dephasing(true)
        .apply(
            &NoiseEvent::IdleTime {
                qubits: &[QubitId(0)],
                duration,
            },
            &mut ctx,
            &mut rng,
        );
}

#[test]
fn builders_enforce_angle_rate_and_duration_bounds() {
    use pecos_neo::noise::validation::{MAX_ANGLE_RATE, MAX_DURATION};
    let rate = MAX_ANGLE_RATE.next_up();
    assert!(NoiseModelBuilder::new().with_idle_noise(0.0, rate).is_err());
    assert!(
        GeneralNoiseModelBuilder::new()
            .with_p_idle_quadratic(rate)
            .is_err()
    );
    assert!(
        CompositeNoiseModelBuilder::new()
            .with_p_idle_quadratic(rate)
            .is_err()
    );
    assert!(
        CompositeNoiseModelBuilder::new()
            .with_idle_noise(0.0, rate)
            .is_err()
    );
    assert!(
        NoiseModelBuilder::new()
            .with_idle_noise(0.0, MAX_ANGLE_RATE)
            .is_ok()
    );
    assert!(
        GeneralNoiseModelBuilder::new()
            .with_p_idle_quadratic(MAX_ANGLE_RATE)
            .is_ok()
    );
    assert!(
        CompositeNoiseModelBuilder::new()
            .with_p_idle_quadratic(MAX_ANGLE_RATE)
            .is_ok()
    );
    for duration in [MAX_DURATION.next_up(), -1.0, f64::NAN, f64::INFINITY] {
        assert!(IdleChannel::default().with_idle_after_2q(duration).is_err());
        assert!(
            NoiseModelBuilder::new()
                .with_idle_after_2q(duration)
                .is_err()
        );
        assert!(
            GeneralNoiseModelBuilder::new()
                .with_idle_after_2q(duration)
                .is_err()
        );
        assert!(
            CompositeNoiseModelBuilder::new()
                .with_idle_after_2q(duration)
                .is_err()
        );
        assert!(
            pecos_neo::extensible::GateNoiseParams::default()
                .with_duration(duration)
                .is_err()
        );
    }
    assert!(
        IdleChannel::default()
            .with_idle_after_2q(MAX_DURATION)
            .is_ok()
    );
    assert!(
        NoiseModelBuilder::new()
            .with_idle_after_2q(MAX_DURATION)
            .is_ok()
    );
    assert!(
        GeneralNoiseModelBuilder::new()
            .with_idle_after_2q(MAX_DURATION)
            .is_ok()
    );
    assert!(
        CompositeNoiseModelBuilder::new()
            .with_idle_after_2q(MAX_DURATION)
            .is_ok()
    );
}
