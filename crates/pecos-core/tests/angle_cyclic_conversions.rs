// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use num_traits::ToPrimitive;
use pecos_core::controlled_rotations::{lower_cphase, lower_crz};
use pecos_core::{Angle8, Angle16, Angle32, Angle64, Angle128, QubitId};
use std::f64::consts::{FRAC_PI_2, PI, TAU};

#[test]
fn tiny_negative_radians_wrap_after_rounding() {
    for (input, distance) in [
        (-1e-15, 2048),
        (-1e-16, 0),
        (-1e-17, 0),
        (-1e-18, 0),
        (-f64::MIN_POSITIVE, 0),
    ] {
        assert_eq!(
            Angle64::from_radians(input),
            -Angle64::new(distance),
            "input={input}"
        );
    }
}

#[test]
fn tiny_negative_turns_wrap_after_rounding() {
    for (input, distance) in [
        (-1e-15, 18_432),
        (-1e-16, 2048),
        (-1e-17, 0),
        (-1e-18, 0),
        (-f64::MIN_POSITIVE, 0),
    ] {
        assert_eq!(
            Angle64::from_turns(input),
            -Angle64::new(distance),
            "input={input}"
        );
    }
}

#[test]
fn rounding_to_full_turn_wraps_even_when_ratio_is_below_one() {
    // This ratio is strictly below one, but rounds to 256 fraction units.
    let turns = 1.0 - 1.0 / 1024.0;
    assert_eq!(Angle8::from_turns(turns), Angle8::ZERO);
    assert_eq!(Angle8::from_radians(turns * TAU), Angle8::ZERO);
    assert_eq!(Angle8::from_turns(1.0 - 1.0 / 512.0), Angle8::ZERO);
    assert_eq!(Angle8::from_turns(1.0 - 1.0 / 256.0), Angle8::new(255));
}

#[test]
fn conversions_agree_with_constants_at_every_width() {
    macro_rules! check {
        ($($angle:ty),+) => {$(
            assert_eq!(<$angle>::from_radians(PI), <$angle>::HALF_TURN);
            assert_eq!(<$angle>::from_radians(FRAC_PI_2), <$angle>::QUARTER_TURN);
            assert_eq!(<$angle>::from_turns(0.75), <$angle>::THREE_QUARTERS_TURN);
            assert_eq!(<$angle>::from_radians(TAU), <$angle>::ZERO);
            assert_eq!(<$angle>::from_turns(1.0), <$angle>::ZERO);
            assert_eq!(<$angle>::HALF_TURN.to_radians().to_bits(), PI.to_bits());
            assert_eq!(<$angle>::THREE_QUARTERS_TURN.to_turns().to_bits(), 0.75_f64.to_bits());
            assert_eq!(format!("{}", <$angle>::THREE_QUARTERS_TURN), "0.750000 turns");
            assert_eq!(<$angle>::from_radians(-1e-18), <$angle>::ZERO);
            assert_eq!(<$angle>::from_turns(-1e-18), <$angle>::ZERO);
        )+};
    }
    check!(Angle8, Angle16, Angle32, Angle64, Angle128);
}

#[test]
fn narrow_angles_round_trip_every_fraction() {
    for fraction in 0..=u16::MAX {
        let angle = Angle16::new(fraction);
        assert_eq!(Angle16::from_turns(angle.to_turns()), angle);
        assert_eq!(Angle16::from_radians(angle.to_radians()), angle);
    }
    for fraction in 0..=u8::MAX {
        let angle = Angle8::new(fraction);
        assert_eq!(Angle8::from_turns(angle.to_turns()), angle);
        assert_eq!(Angle8::from_radians(angle.to_radians()), angle);
    }

    macro_rules! check_last_interval {
        ($angle:ty, $integer:ty, $modulus:expr) => {
            for (offset, expected) in [
                (0.75, <$angle>::new(<$integer>::MAX)),
                (0.25, <$angle>::ZERO),
                (0.125, <$angle>::ZERO),
            ] {
                let turns = 1.0 - offset / $modulus;
                assert!(turns > <$angle>::new(<$integer>::MAX).to_turns());
                assert!(turns < 1.0);
                let from_turns = <$angle>::from_turns(turns);
                let from_radians = <$angle>::from_radians(turns * TAU);
                assert_eq!(from_turns.fraction(), expected.fraction());
                assert_eq!(from_radians.fraction(), expected.fraction());
                assert_eq!(<$angle>::from_turns(from_turns.to_turns()), expected);
                assert_eq!(<$angle>::from_radians(from_radians.to_radians()), expected);
            }
        };
    }
    check_last_interval!(Angle8, u8, 256.0);
    check_last_interval!(Angle16, u16, 65_536.0);
}

#[test]
fn radians_round_trip_across_wrap_point() {
    // Close to the wrap, a broad radians tolerance would also accept zero.
    for (input, expected) in [
        (-TAU.next_up(), -Angle64::new(2048)),
        (-TAU, Angle64::ZERO),
        (-TAU.next_down(), Angle64::new(2608)),
        (0.0, Angle64::ZERO),
        (TAU.next_down(), -Angle64::new(2048)),
        (TAU, Angle64::ZERO),
        (TAU.next_up(), Angle64::new(2608)),
    ] {
        assert_eq!(Angle64::from_radians(input).fraction(), expected.fraction());
    }
    for input in [-100.0, -PI, -0.7, 0.7, FRAC_PI_2, PI, 100.0] {
        let actual = Angle64::from_radians(input).to_radians();
        let expected = input.rem_euclid(TAU);
        // Compare on the circle: a rounded full turn is zero.
        let difference = (actual - expected).abs();
        assert!(
            difference.min((TAU - difference).abs()) <= 2.0 * f64::EPSILON * TAU,
            "input={input}, actual={actual}, expected={expected}"
        );
    }
}

#[test]
fn tiny_positive_constructors_have_exact_fractions() {
    for (input, radians_fraction, turns_fraction) in [
        (1e-15, 2936, 18_447),
        (1e-16, 294, 1845),
        (1e-17, 29, 184),
        (1e-18, 3, 18),
        (f64::MIN_POSITIVE, 0, 0),
    ] {
        assert_eq!(Angle64::from_radians(input).fraction(), radians_fraction);
        assert_eq!(Angle64::from_turns(input).fraction(), turns_fraction);
    }
}

#[test]
fn constructors_handle_subnormals_negative_zero_and_largest_finite_inputs() {
    let smallest_subnormal = f64::from_bits(1);
    let largest_subnormal = f64::MIN_POSITIVE.next_down();
    assert!(smallest_subnormal.is_subnormal());
    assert!(largest_subnormal.is_subnormal());
    macro_rules! check {
        ($angle:ty, $positive_max:expr, $negative_max:expr) => {
            for input in [
                smallest_subnormal,
                -smallest_subnormal,
                largest_subnormal,
                -largest_subnormal,
                -0.0,
            ] {
                assert_eq!(<$angle>::from_radians(input).fraction(), 0);
                assert_eq!(<$angle>::from_turns(input).fraction(), 0);
            }
            assert_eq!(<$angle>::from_radians(f64::MAX).fraction(), $positive_max);
            assert_eq!(<$angle>::from_radians(-f64::MAX).fraction(), $negative_max);
            assert_eq!(<$angle>::from_turns(f64::MAX).fraction(), 0);
            assert_eq!(<$angle>::from_turns(-f64::MAX).fraction(), 0);
        };
    }
    check!(Angle8, 24, 232);
    check!(Angle16, 6056, 59_480);
    check!(Angle32, 396_914_332, 3_898_052_964);
    check!(
        Angle64,
        1_704_734_075_010_201_088,
        16_742_009_998_699_350_016
    );
    check!(
        Angle128,
        31_446_793_195_445_161_152_375_904_822_475_358_208,
        308_835_573_725_493_292_866_265_736_870_002_425_856
    );
}

#[test]
fn wide_angle_outputs_are_bit_identical_to_previous_scale() {
    macro_rules! check {
        ($angle:ty, $integer:ty, $fraction:expr) => {{
            let fraction: $integer = $fraction;
            let old_scale = <$integer>::MAX.to_f64().unwrap();
            assert_eq!(old_scale.to_bits(), (old_scale + 1.0).to_bits());
            let old_turns = fraction.to_f64().unwrap() / old_scale;
            let angle = <$angle>::new(fraction);
            assert_eq!(angle.to_turns().to_bits(), old_turns.to_bits());
            assert_eq!(angle.to_radians().to_bits(), (old_turns * TAU).to_bits());
        }};
    }
    for fraction in [
        0,
        1,
        (1 << 62) - 1,
        1 << 62,
        1 << 63,
        u64::MAX - 1,
        u64::MAX,
    ] {
        check!(Angle64, u64, fraction);
        check!(Angle128, u128, u128::from(fraction) << 64);
    }
    check!(Angle128, u128, u128::MAX);
    let mut fraction = 1_u64;
    for _ in 0..100_000 {
        fraction = fraction
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        check!(Angle64, u64, fraction);
        check!(
            Angle128,
            u128,
            (u128::from(fraction) << 64) | u128::from(!fraction)
        );
    }
}

#[test]
fn tolerance_uses_fraction_units_but_clamps_full_turns() {
    assert_eq!(Angle8::epsilon_from_turns(0.75), 192);
    assert_eq!(Angle8::epsilon_from_radians(0.75 * TAU), 192);
    macro_rules! check {
        ($angle:ty, $integer:ty) => {
            for turns in [1.0, 2.0, -1.0, -2.0] {
                assert_eq!(<$angle>::epsilon_from_turns(turns), <$integer>::MAX);
                assert_eq!(<$angle>::epsilon_from_radians(turns * TAU), <$integer>::MAX);
            }
            for input in [f64::MAX, -f64::MAX] {
                assert_eq!(<$angle>::epsilon_from_turns(input), <$integer>::MAX);
                assert_eq!(<$angle>::epsilon_from_radians(input), <$integer>::MAX);
            }
            for input in [0.0, -0.0, f64::from_bits(1), -f64::from_bits(1)] {
                assert_eq!(<$angle>::epsilon_from_turns(input), 0);
                assert_eq!(<$angle>::epsilon_from_radians(input), 0);
            }
            assert_eq!(
                <$angle>::epsilon_from_turns(0.5),
                <$angle>::HALF_TURN.fraction()
            );
            assert_eq!(
                <$angle>::epsilon_from_radians(PI),
                <$angle>::HALF_TURN.fraction()
            );
            assert!(<$angle>::ZERO.abs_diff_eq_turns(&<$angle>::HALF_TURN, 1.0));
            assert!(<$angle>::ZERO.abs_diff_eq_radians(&<$angle>::HALF_TURN, TAU));
        };
    }
    check!(Angle8, u8);
    check!(Angle16, u16);
    check!(Angle32, u32);
    check!(Angle64, u64);
    check!(Angle128, u128);
}

#[test]
fn tolerance_preserves_values_just_below_saturation() {
    macro_rules! check {
        ($angle:ty, $expected:expr) => {
            assert_eq!(<$angle>::epsilon_from_turns(1.0_f64.next_down()), $expected);
            assert_eq!(<$angle>::epsilon_from_radians(TAU.next_down()), $expected);
        };
    }
    check!(Angle8, u8::MAX);
    check!(Angle16, u16::MAX);
    check!(Angle32, u32::MAX);
    check!(Angle64, u64::MAX - 2047);
    check!(Angle128, u128::MAX - ((1_u128 << 75) - 1));
}

#[test]
fn nonfinite_conversions_panic_and_infinite_tolerances_saturate() {
    macro_rules! check {
        ($($angle:ty),+) => {$(
            for input in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                assert!(std::panic::catch_unwind(|| <$angle>::from_radians(input)).is_err());
                assert!(std::panic::catch_unwind(|| <$angle>::from_turns(input)).is_err());
            }
            assert!(std::panic::catch_unwind(|| <$angle>::epsilon_from_radians(f64::NAN)).is_err());
            assert!(std::panic::catch_unwind(|| <$angle>::epsilon_from_turns(f64::NAN)).is_err());
        )+};
    }
    check!(Angle8, Angle16, Angle32, Angle64, Angle128);

    // An infinite tolerance saturates at every width, for the same reason a
    // full turn does: every angle lies within it. This was previously
    // width-dependent, saturating only where `max_value` was exactly
    // representable in f64 and panicking at `u64` and `u128`.
    macro_rules! saturates {
        ($($angle:ty => $integer:ty),+ $(,)?) => {$(
            for input in [f64::INFINITY, f64::NEG_INFINITY] {
                assert_eq!(<$angle>::epsilon_from_radians(input), <$integer>::MAX);
                assert_eq!(<$angle>::epsilon_from_turns(input), <$integer>::MAX);
            }
        )+};
    }
    saturates!(
        Angle8 => u8,
        Angle16 => u16,
        Angle32 => u32,
        Angle64 => u64,
        Angle128 => u128,
    );
}

#[test]
fn tiny_controlled_rotations_have_expected_angles() {
    // Reduction before halving leaves positive inputs unchanged. For negative
    // inputs, adding TAU rounds -1e-15 to -8.881784197001252e-16 and smaller
    // magnitudes to zero; the stored half therefore need not be symmetric.
    for (theta, rzz_angle, rz_angle) in [
        (1e-15, -Angle64::new(2048), Angle64::new(1468)),
        (-1e-15, Angle64::new(1304), Angle64::ZERO),
        (1e-16, Angle64::ZERO, Angle64::new(147)),
        (-1e-16, Angle64::ZERO, Angle64::ZERO),
        (1e-17, Angle64::ZERO, Angle64::new(15)),
        (-1e-17, Angle64::ZERO, Angle64::ZERO),
        (1e-18, Angle64::ZERO, Angle64::new(1)),
        (-1e-18, Angle64::ZERO, Angle64::ZERO),
        (f64::MIN_POSITIVE, Angle64::ZERO, Angle64::ZERO),
        (-f64::MIN_POSITIVE, Angle64::ZERO, Angle64::ZERO),
    ] {
        let gates = lower_crz(theta, QubitId(0), QubitId(1));
        let [rzz, rz] = gates.as_slice() else {
            panic!("tiny angles must not need a control Z");
        };
        assert_eq!(rzz.angles.as_slice(), &[rzz_angle], "theta={theta}");
        assert_eq!(rz.angles.as_slice(), &[rz_angle], "theta={theta}");
    }
}

#[test]
fn tiny_controlled_phases_have_expected_angles() {
    for (input, rzz_angle, half_angle) in [
        (1e-15, -Angle64::new(2048), Angle64::new(1468)),
        (-1e-15, Angle64::new(1304), Angle64::ZERO),
        (1e-16, Angle64::ZERO, Angle64::new(147)),
        (-1e-16, Angle64::ZERO, Angle64::ZERO),
        (1e-17, Angle64::ZERO, Angle64::new(15)),
        (-1e-17, Angle64::ZERO, Angle64::ZERO),
        (1e-18, Angle64::ZERO, Angle64::new(2)),
        (-1e-18, Angle64::ZERO, Angle64::ZERO),
        (f64::MIN_POSITIVE, Angle64::ZERO, Angle64::ZERO),
        (-f64::MIN_POSITIVE, Angle64::ZERO, Angle64::ZERO),
    ] {
        let [rzz, u, rz] = lower_cphase(input, QubitId(0), QubitId(1));
        assert_eq!(rzz.angles.as_slice(), &[rzz_angle], "input={input}");
        assert_eq!(
            u.angles.as_slice(),
            &[Angle64::ZERO, Angle64::ZERO, half_angle],
            "input={input}"
        );
        assert_eq!(rz.angles.as_slice(), &[half_angle], "input={input}");
    }
}
