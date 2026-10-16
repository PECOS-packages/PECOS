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
use pecos_core::controlled_rotations::{lower_cphase, lower_crz, lower_phase};
use pecos_core::{Angle8, Angle16, Angle32, Angle64, Angle128, QubitId};
use std::f64::consts::{FRAC_PI_2, PI, TAU};

#[test]
fn tiny_negative_radians_round_magnitude_then_wrap() {
    for (input, distance) in [
        (-1e-15, 2936),
        (-1e-16, 294),
        (-1e-17, 29),
        (-1e-18, 3),
        (-f64::MIN_POSITIVE, 0),
    ] {
        assert_eq!(
            Angle64::try_from_radians(input).unwrap(),
            -Angle64::new(distance),
            "input={input}"
        );
    }
}

#[test]
fn tiny_negative_turns_round_magnitude_then_wrap() {
    for (input, distance) in [
        (-1e-15, 18_447),
        (-1e-16, 1845),
        (-1e-17, 184),
        (-1e-18, 18),
        (-f64::MIN_POSITIVE, 0),
    ] {
        assert_eq!(
            Angle64::try_from_turns(input).unwrap(),
            -Angle64::new(distance),
            "input={input}"
        );
    }
}

#[test]
fn rounding_to_full_turn_wraps_even_when_ratio_is_below_one() {
    // This ratio is strictly below one, but rounds to 256 fraction units.
    let turns = 1.0 - 1.0 / 1024.0;
    assert_eq!(Angle8::try_from_turns(turns).unwrap(), Angle8::ZERO);
    assert_eq!(Angle8::try_from_radians(turns * TAU).unwrap(), Angle8::ZERO);
    assert_eq!(
        Angle8::try_from_turns(std::hint::black_box(1.0 - 1.0 / 512.0)).unwrap(),
        Angle8::ZERO
    );
    assert_eq!(
        Angle8::try_from_turns(std::hint::black_box(1.0 - 1.0 / 256.0)).unwrap(),
        Angle8::new(255)
    );
}

#[test]
fn conversions_agree_with_constants_at_every_width() {
    macro_rules! check {
        ($($angle:ty),+) => {$(
            assert_eq!(<$angle>::try_from_radians(std::hint::black_box(PI)).unwrap(), <$angle>::HALF_TURN);
            assert_eq!(<$angle>::try_from_radians(std::hint::black_box(FRAC_PI_2)).unwrap(), <$angle>::QUARTER_TURN);
            assert_eq!(<$angle>::try_from_turns(std::hint::black_box(0.75)).unwrap(), <$angle>::THREE_QUARTERS_TURN);
            assert_eq!(<$angle>::try_from_radians(std::hint::black_box(TAU)).unwrap(), <$angle>::ZERO);
            assert_eq!(<$angle>::try_from_turns(std::hint::black_box(1.0)).unwrap(), <$angle>::ZERO);
            assert_eq!(<$angle>::HALF_TURN.to_radians().to_bits(), PI.to_bits());
            assert_eq!(<$angle>::THREE_QUARTERS_TURN.to_turns().to_bits(), 0.75_f64.to_bits());
            assert_eq!(format!("{}", <$angle>::THREE_QUARTERS_TURN), "0.750000 turns");
            assert_eq!(<$angle>::try_from_radians(std::hint::black_box(-1e-18)).unwrap(), -<$angle>::try_from_radians(1e-18).unwrap());
            assert_eq!(<$angle>::try_from_turns(std::hint::black_box(-1e-18)).unwrap(), -<$angle>::try_from_turns(1e-18).unwrap());
        )+};
    }
    check!(Angle8, Angle16, Angle32, Angle64, Angle128);
}

#[test]
fn narrow_angles_round_trip_every_fraction() {
    for fraction in 0..=u16::MAX {
        let angle = Angle16::new(fraction);
        assert_eq!(Angle16::try_from_turns(angle.to_turns()).unwrap(), angle);
        assert_eq!(
            Angle16::try_from_radians(angle.to_radians()).unwrap(),
            angle
        );
    }
    for fraction in 0..=u8::MAX {
        let angle = Angle8::new(fraction);
        assert_eq!(Angle8::try_from_turns(angle.to_turns()).unwrap(), angle);
        assert_eq!(Angle8::try_from_radians(angle.to_radians()).unwrap(), angle);
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
                let from_turns = <$angle>::try_from_turns(turns).unwrap();
                let from_radians = <$angle>::try_from_radians(turns * TAU).unwrap();
                assert_eq!(from_turns.fraction(), expected.fraction());
                assert_eq!(from_radians.fraction(), expected.fraction());
                assert_eq!(
                    <$angle>::try_from_turns(from_turns.to_turns()).unwrap(),
                    expected
                );
                assert_eq!(
                    <$angle>::try_from_radians(from_radians.to_radians()).unwrap(),
                    expected
                );
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
        (-TAU.next_up(), -Angle64::new(2608)),
        (-TAU, Angle64::ZERO),
        (-TAU.next_down(), Angle64::new(2608)),
        (0.0, Angle64::ZERO),
        (TAU.next_down(), -Angle64::new(2608)),
        (TAU, Angle64::ZERO),
        (TAU.next_up(), Angle64::new(2608)),
    ] {
        assert_eq!(
            Angle64::try_from_radians(input).unwrap().fraction(),
            expected.fraction()
        );
    }
    for input in [-100.0, -PI, -0.7, 0.7, FRAC_PI_2, PI, 100.0] {
        let actual = Angle64::try_from_radians(input).unwrap().to_radians();
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
        assert_eq!(
            Angle64::try_from_radians(input).unwrap().fraction(),
            radians_fraction
        );
        assert_eq!(
            Angle64::try_from_turns(input).unwrap().fraction(),
            turns_fraction
        );
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
                assert_eq!(<$angle>::try_from_radians(input).unwrap().fraction(), 0);
                assert_eq!(<$angle>::try_from_turns(input).unwrap().fraction(), 0);
            }
            assert_eq!(
                <$angle>::try_from_radians(std::hint::black_box(f64::MAX))
                    .unwrap()
                    .fraction(),
                $positive_max
            );
            assert_eq!(
                <$angle>::try_from_radians(std::hint::black_box(-f64::MAX))
                    .unwrap()
                    .fraction(),
                $negative_max
            );
            assert_eq!(
                <$angle>::try_from_turns(std::hint::black_box(f64::MAX))
                    .unwrap()
                    .fraction(),
                0
            );
            assert_eq!(
                <$angle>::try_from_turns(std::hint::black_box(-f64::MAX))
                    .unwrap()
                    .fraction(),
                0
            );
        };
    }
    check!(Angle8, 24, 232);
    check!(Angle16, 6056, 59_480);
    check!(Angle32, 396_914_332, 3_898_052_964);
    check!(
        Angle64,
        1_704_734_075_010_201_102,
        16_742_009_998_699_350_514
    );
    check!(
        Angle128,
        31_446_793_195_445_161_409_785_512_896_274_356_550,
        308_835_573_725_493_302_053_589_094_535_493_854_906
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
    assert_eq!(Angle8::try_epsilon_from_turns(0.75).unwrap(), 192);
    assert_eq!(Angle8::try_epsilon_from_radians(0.75 * TAU).unwrap(), 192);
    macro_rules! check {
        ($angle:ty, $integer:ty) => {
            for turns in [1.0, 2.0, -1.0, -2.0] {
                assert_eq!(
                    <$angle>::try_epsilon_from_turns(turns).unwrap(),
                    <$integer>::MAX
                );
                assert_eq!(
                    <$angle>::try_epsilon_from_radians(turns * TAU).unwrap(),
                    <$integer>::MAX
                );
            }
            for input in [f64::MAX, -f64::MAX] {
                assert_eq!(
                    <$angle>::try_epsilon_from_turns(input).unwrap(),
                    <$integer>::MAX
                );
                assert_eq!(
                    <$angle>::try_epsilon_from_radians(input).unwrap(),
                    <$integer>::MAX
                );
            }
            for input in [0.0, -0.0, f64::from_bits(1), -f64::from_bits(1)] {
                assert_eq!(<$angle>::try_epsilon_from_turns(input).unwrap(), 0);
                assert_eq!(<$angle>::try_epsilon_from_radians(input).unwrap(), 0);
            }
            assert_eq!(
                <$angle>::try_epsilon_from_turns(0.5).unwrap(),
                <$angle>::HALF_TURN.fraction()
            );
            assert_eq!(
                <$angle>::try_epsilon_from_radians(PI).unwrap(),
                <$angle>::HALF_TURN.fraction()
            );
            assert!(<$angle>::ZERO.abs_diff_eq(
                &<$angle>::HALF_TURN,
                <$angle>::try_epsilon_from_turns(1.0).unwrap()
            ));
            assert!(<$angle>::ZERO.abs_diff_eq(
                &<$angle>::HALF_TURN,
                <$angle>::try_epsilon_from_radians(TAU).unwrap()
            ));
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
            assert_eq!(
                <$angle>::try_epsilon_from_turns(1.0_f64.next_down()).unwrap(),
                $expected
            );
            assert_eq!(
                <$angle>::try_epsilon_from_radians(TAU.next_down()).unwrap(),
                $expected
            );
        };
    }
    check!(Angle8, u8::MAX);
    check!(Angle16, u16::MAX);
    check!(Angle32, u32::MAX);
    check!(Angle64, u64::MAX - 2047);
    check!(Angle128, u128::MAX - ((1_u128 << 75) - 1));
}

#[test]
fn nonfinite_conversions_error_and_infinite_tolerances_saturate() {
    macro_rules! check {
        ($($angle:ty),+) => {$(
            for input in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                assert!(<$angle>::try_from_radians(input).is_err());
                assert!(<$angle>::try_from_turns(input).is_err());
            }
            assert!(<$angle>::try_epsilon_from_radians(f64::NAN).is_err());
            assert!(<$angle>::try_epsilon_from_turns(f64::NAN).is_err());
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
                assert_eq!(<$angle>::try_epsilon_from_radians(input).unwrap(), <$integer>::MAX);
                assert_eq!(<$angle>::try_epsilon_from_turns(input).unwrap(), <$integer>::MAX);
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
    // Halve before quantizing once to the nearest tick, then negate in fixed point.
    // Both signs retain tiny residuals and the legs remain exact negatives.
    for (theta, rzz_angle, rz_angle) in [
        (1e-15, -Angle64::new(1468), Angle64::new(1468)),
        (-1e-15, Angle64::new(1468), -Angle64::new(1468)),
        (1e-16, -Angle64::new(147), Angle64::new(147)),
        (-1e-16, Angle64::new(147), -Angle64::new(147)),
        (1e-17, -Angle64::new(15), Angle64::new(15)),
        (-1e-17, Angle64::new(15), -Angle64::new(15)),
        (1e-18, -Angle64::new(1), Angle64::new(1)),
        (-1e-18, Angle64::new(1), -Angle64::new(1)),
        (f64::MIN_POSITIVE, Angle64::ZERO, Angle64::ZERO),
        (-f64::MIN_POSITIVE, Angle64::ZERO, Angle64::ZERO),
    ] {
        let gates = lower_crz(theta, QubitId(0), QubitId(1)).unwrap();
        let [rzz, rz] = gates.as_slice() else {
            panic!("tiny angles must not need a control Z");
        };
        assert_eq!(rzz.angles.as_slice(), &[rzz_angle], "theta={theta}");
        assert_eq!(rz.angles.as_slice(), &[rz_angle], "theta={theta}");
    }
}

#[test]
fn tiny_controlled_phases_have_expected_angles() {
    // The floating-point entry point quantizes the half once to the nearest tick;
    // RZZ is its exact integer negation, as in CRZ.
    for (input, half_angle) in [
        (1e-15, Angle64::new(1468)),
        (-1e-15, -Angle64::new(1468)),
        (1e-16, Angle64::new(147)),
        (-1e-16, -Angle64::new(147)),
        (1e-17, Angle64::new(15)),
        (-1e-17, -Angle64::new(15)),
        (1e-18, Angle64::new(1)),
        (-1e-18, -Angle64::new(1)),
        (f64::MIN_POSITIVE, Angle64::ZERO),
        (-f64::MIN_POSITIVE, Angle64::ZERO),
    ] {
        let [rzz, u, rz] = lower_cphase(input, QubitId(0), QubitId(1)).unwrap();
        assert_eq!(rzz.angles.as_slice(), &[-half_angle], "input={input}");
        assert_eq!(
            u.angles.as_slice(),
            &[Angle64::ZERO, Angle64::ZERO, half_angle],
            "input={input}"
        );
        assert_eq!(rz.angles.as_slice(), &[half_angle], "input={input}");
    }
}

#[test]
fn decimal_pi_overflow_is_an_error() {
    let text = format!("{}{}.0pi", "1", "0".repeat(308));
    assert!(matches!(
        Angle64::from_str_radians(&text),
        Err(pecos_core::ParseAngleError::DecimalOverflow { .. })
    ));
}

#[test]
fn scientific_notation_is_rejected_by_integer_parser() {
    assert!(matches!(
        Angle64::from_str_radians("1e400"),
        Err(pecos_core::ParseAngleError::InvalidFormat)
    ));
}
#[test]
fn nonfinite_angles_are_errors() {
    macro_rules! check {
        ($angle:ty) => {{
            for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let error = <$angle>::try_from_radians(value).unwrap_err();
                assert!(error.to_string().contains(&value.to_string()));
                assert!(error.to_string().contains("radians"));
                let error = <$angle>::try_from_turns(value).unwrap_err();
                assert!(error.to_string().contains(&value.to_string()));
                assert!(error.to_string().contains("turns"));
            }
            for value in [0.0, -1.0, 1e300, -1e300, f64::MAX, f64::MIN] {
                assert!(<$angle>::try_from_radians(value).is_ok());
                assert!(<$angle>::try_from_turns(value).is_ok());
            }
        }};
    }
    check!(Angle8);
    check!(Angle16);
    check!(Angle32);
    check!(Angle64);
    check!(Angle128);
}

#[test]
fn precise_conversions_preserve_sign_and_constants_at_every_width() {
    macro_rules! check {
        ($angle:ty, $bits:expr) => {{
            let tick = 2.0_f64.powi(-$bits);
            for value in [
                0.0,
                tick * 0.5,
                tick * 1.5,
                tick * 2.5,
                tick * 3.0,
                1e-18,
                1e-17,
                1e-16,
                1e-15,
                0.125,
                0.37,
                0.5,
                0.75,
                1.0_f64.next_down(),
                100.0,
                1e6,
                f64::MAX,
            ] {
                assert_eq!(
                    <$angle>::try_from_turns(-value).unwrap(),
                    -<$angle>::try_from_turns(value).unwrap(),
                    "turns={value}, bits={}",
                    $bits
                );
                assert_eq!(
                    <$angle>::try_from_radians(-value).unwrap(),
                    -<$angle>::try_from_radians(value).unwrap(),
                    "radians={value}, bits={}",
                    $bits
                );
            }
            for (numerator, denominator, turns) in [
                (1, 16, 0.0625),
                (1, 8, 0.125),
                (1, 4, 0.25),
                (3, 8, 0.375),
                (1, 2, 0.5),
                (3, 4, 0.75),
            ] {
                let exact = <$angle>::from_turn_ratio(numerator, denominator);
                assert_eq!(<$angle>::try_from_turns(turns).unwrap(), exact);
                assert_eq!(<$angle>::try_from_turns(-turns).unwrap(), -exact);
                assert_eq!(<$angle>::try_from_radians(turns * TAU).unwrap(), exact);
                assert_eq!(<$angle>::try_from_radians(-turns * TAU).unwrap(), -exact);
            }
            assert_eq!(
                <$angle>::try_from_turns(0.5 * tick).unwrap(),
                <$angle>::new(1)
            );
            assert_eq!(
                <$angle>::try_from_turns(-0.5 * tick).unwrap(),
                -<$angle>::new(1)
            );
        }};
    }
    check!(Angle8, 8);
    check!(Angle16, 16);
    check!(Angle32, 32);
    check!(Angle64, 64);
    check!(Angle128, 128);
}

#[test]
fn floating_controlled_halves_are_nearest_tick() {
    for input in [
        1e-17_f64,
        1e-15,
        1e-16,
        1e-18,
        f64::MIN_POSITIVE,
        f64::from_bits(1),
        f64::from_bits(3),
    ] {
        for sign in [-1.0, 1.0] {
            let value = sign * input;
            let crz = lower_crz(value, QubitId(0), QubitId(1)).unwrap();
            let cp = lower_cphase(value, QubitId(0), QubitId(1)).unwrap();
            assert_eq!(lower_phase(value, &[QubitId(0), QubitId(1)]).unwrap(), cp);
            for (name, half) in [
                ("CRZ", crz.last().unwrap().angles[0]),
                ("CPhase", cp[2].angles[0]),
            ] {
                let magnitude = if value < 0.0 { -half } else { half };
                let ticks = magnitude.fraction().to_f64().unwrap();
                let ideal = (input / 2.0 / TAU) * 2.0_f64.powi(64);
                assert!(
                    (ticks - ideal).abs() <= 0.5,
                    "{name} input={value}, actual={ticks}, ideal={ideal}"
                );
            }
            assert_eq!(crz[0].angles[0], -crz.last().unwrap().angles[0]);
            assert_eq!(cp[0].angles[0], -cp[2].angles[0]);
        }
    }
}

#[test]
fn integer_reduction_handles_wide_boundaries() {
    macro_rules! check {
        ($angle:ty, $bits:expr) => {{
            // Very large f64 turn values have no fractional bits remaining.
            // Signed remainder must produce zero on either side of each boundary.
            let limit = 2.0_f64.powi(127 - $bits);
            for turns in [limit.next_down(), limit, limit.next_up()] {
                for sign in [-1.0, 1.0] {
                    assert_eq!(
                        <$angle>::try_from_turns(sign * turns).unwrap(),
                        <$angle>::ZERO
                    );
                }
            }
            // Check integer reduction after many complete turns, including signs.
            assert_eq!(
                <$angle>::try_from_turns(1_048_576.25).unwrap(),
                <$angle>::QUARTER_TURN
            );
            assert_eq!(
                <$angle>::try_from_turns(-1_048_576.25).unwrap(),
                <$angle>::THREE_QUARTERS_TURN
            );
        }};
    }
    check!(Angle8, 8);
    check!(Angle16, 16);
    check!(Angle32, 32);
    check!(Angle64, 64);

    // u128 requires a sign plus an unsigned magnitude above half a turn.
    // The predecessor of one is exactly 1 - 2^-53; reducing either sign must
    // preserve that distance on both sides of the full-turn boundary.
    let distance = Angle128::new(1_u128 << 75);
    assert_eq!(
        Angle128::try_from_turns(1.0_f64.next_down()).unwrap(),
        -distance
    );
    assert_eq!(
        Angle128::try_from_turns(-1.0_f64.next_down()).unwrap(),
        distance
    );
    assert_eq!(Angle128::try_from_turns(1.0).unwrap(), Angle128::ZERO);
    assert_eq!(Angle128::try_from_turns(-1.0).unwrap(), Angle128::ZERO);
    assert_eq!(
        Angle128::try_from_turns(1.0_f64.next_up()).unwrap(),
        distance + distance
    );
    assert_eq!(
        Angle128::try_from_turns(-1.0_f64.next_up()).unwrap(),
        -(distance + distance)
    );
}

#[test]
fn signed_remainder_radians_are_correctly_rounded() {
    for (magnitude, expected_new_error, expected_old_errors) in
        [(100.0_f64, 0, [562, 50]), (1e6, 0, [36, 36])]
    {
        let mut new_errors = Vec::new();
        let mut old_errors = Vec::new();
        for (sign, expected_old_error) in [1.0, -1.0].into_iter().zip(expected_old_errors) {
            let value = sign * magnitude;
            let remainder = value % TAU;
            // These remainders and TAU share a binary exponent. Their exact
            // ratio is the ratio of their 53-bit integer significands. Scaling
            // its numerator by 2^64 fits u128: the oracle uses no f64 division.
            assert_eq!(
                (remainder.abs().to_bits() >> 52) & 0x7ff,
                (TAU.to_bits() >> 52) & 0x7ff
            );
            let significand =
                |x: f64| u128::from((x.to_bits() & ((1_u64 << 52) - 1)) | (1_u64 << 52));
            let denominator = significand(TAU);
            let exact_numerator = significand(remainder.abs()) << 64;
            let exact_ticks = (exact_numerator + denominator / 2) / denominator;
            let exact = Angle64::new(u64::try_from(exact_ticks).unwrap());
            let exact = if value < 0.0 { -exact } else { exact };
            let modulus = 2.0_f64.powi(64);
            let old_ticks = ((value.rem_euclid(TAU) / TAU) * modulus)
                .round()
                .rem_euclid(modulus);
            let old = Angle64::new(old_ticks.to_u64().unwrap());
            let actual = Angle64::try_from_radians(value).unwrap();
            assert_eq!(actual.abs_diff(&exact), expected_new_error, "value={value}");
            assert_eq!(old.abs_diff(&exact), expected_old_error, "value={value}");
            // Compare errors to the exact rational, not just its rounded tick.
            let error = |angle: Angle64| {
                let magnitude = if value < 0.0 { -angle } else { angle };
                (u128::from(magnitude.fraction()) * denominator).abs_diff(exact_numerator)
            };
            new_errors.push(error(actual));
            old_errors.push(error(old));
        }
        assert_eq!(new_errors[0], new_errors[1]);
        // Exact rational rounding improves or matches each input separately,
        // including negative inputs where the old unsigned reduction was closer.
        for (new, old) in new_errors.iter().zip(&old_errors) {
            assert!(new <= old);
        }
    }
}

#[test]
fn angle128_turn_ratios_are_exact() {
    assert_eq!(Angle128::from_turn_ratio(1, 4), Angle128::QUARTER_TURN);
    assert_eq!(Angle128::from_turn_ratio(1, 2), Angle128::HALF_TURN);
    assert_eq!(
        Angle128::from_turn_ratio(3, 4),
        Angle128::THREE_QUARTERS_TURN
    );
    assert_eq!(
        Angle128::from_turn_ratio(1, 3),
        Angle128::new(u128::MAX / 3)
    );
    assert_eq!(
        Angle128::from_turn_ratio(1, 7),
        Angle128::new(u128::MAX / 7 + 1)
    );
    assert_eq!(
        Angle128::from_turn_ratio(-1, 8),
        -Angle128::new(1_u128 << 125)
    );
    let gap = Angle128::new((1_u128 << 65) + 4);
    assert_eq!(Angle128::from_turn_ratio(i64::MAX - 1, i64::MAX), -gap);
    assert_eq!(Angle128::from_turn_ratio(1 - i64::MAX, i64::MAX), gap);
    assert_eq!(Angle128::from_turn_ratio(1, i64::MAX), gap);
    for (n, d, expected) in [
        (0, 1, Angle128::ZERO),
        (1, 1, Angle128::ZERO),
        (-1, 1, Angle128::ZERO),
        (1, -8, -Angle128::new(1_u128 << 125)),
        (-1, -8, Angle128::new(1_u128 << 125)),
    ] {
        assert_eq!(Angle128::from_turn_ratio(n, d), expected);
    }
}

#[test]
fn turn_ratio_narrow_values_and_invalid_inputs_are_unchanged() {
    macro_rules! check {
        ($angle:ty, $bits:expr) => {{
            for (numerator, denominator) in [
                (1_i64, 3_i64),
                (1, 7),
                (1, 8),
                (3, 4),
                (i64::MAX / 3, i64::MAX),
            ] {
                let scale = 1_u128 << $bits;
                let magnitude = (u128::from(numerator.unsigned_abs()) * scale
                    + u128::from(denominator.unsigned_abs()) / 2)
                    / u128::from(denominator.unsigned_abs());
                let expected = <$angle>::new(magnitude.try_into().unwrap());
                assert_eq!(<$angle>::from_turn_ratio(numerator, denominator), expected);
                assert_eq!(
                    <$angle>::from_turn_ratio(-numerator, denominator),
                    -expected
                );
                assert_eq!(
                    <$angle>::from_turn_ratio(numerator, -denominator),
                    -expected
                );
                assert_eq!(
                    <$angle>::from_turn_ratio(-numerator, -denominator),
                    expected
                );
            }
        }};
    }
    check!(Angle8, 8);
    check!(Angle16, 16);
    check!(Angle32, 32);
    check!(Angle64, 64);
    macro_rules! invalid {
        ($($angle:ty),+) => {$(
            for (n, d) in [(1, 0), (i64::MAX, 1), (i64::MIN, -1), (1, i64::MIN)] {
                assert!(std::panic::catch_unwind(|| <$angle>::from_turn_ratio(n, d)).is_err());
            }
        )+};
    }
    invalid!(Angle8, Angle16, Angle32, Angle64, Angle128);
}
