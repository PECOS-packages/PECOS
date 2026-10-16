// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

#[path = "support/exact_angle.rs"]
mod exact_angle;
use pecos_core::{Angle8, Angle16, Angle32, Angle64, Angle128};
use std::f64::consts::TAU;

#[test]
fn every_width_matches_exact_rational_oracle() {
    let mut inputs = vec![
        0.0,
        f64::from_bits(1),
        f64::from_bits(3),
        f64::MIN_POSITIVE.next_down(),
        f64::MIN_POSITIVE,
        f64::MAX,
        100.0,
        1e6,
        1e-15,
        0.73,
    ];
    for k in [1.0, 2.0, 4.0, 1e3, 1e15, 1e100, 1e300] {
        let boundary = k * TAU;
        inputs.extend([boundary.next_down(), boundary, boundary.next_up()]);
    }
    for width in [8, 16, 32, 64, 128] {
        for n in [0.5, 1.5, 2.5] {
            let tie = n * 2.0_f64.powi(-width);
            inputs.extend([tie.next_down(), tie, tie.next_up(), tie * TAU]);
        }
        inputs.push(1.0 - 2.0_f64.powi(-width - 1));
    }
    let mut state = 0x2718_2818_2845_9045_u64;
    for _ in 0..1024 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let value = f64::from_bits(state & (u64::MAX >> 1));
        if value.is_finite() {
            inputs.push(value);
        }
    }
    for (period, turns) in [(TAU, false), (1.0, true)] {
        macro_rules! check {
            ($angle:ty, $width:expr) => {{
                for sign in [1.0, -1.0] {
                    let mut max_error = 0.0_f64;
                    for &input in &inputs {
                        let value = sign * input;
                        let (expected, error) = exact_angle::expected(value, period, $width);
                        let actual = if turns { <$angle>::try_from_turns(value) } else { <$angle>::try_from_radians(value) }.unwrap();
                        assert_eq!(u128::from(actual.fraction()), expected, "width={} period={period} value={value:?}", $width);
                        max_error = max_error.max(error);
                    }
                    println!("MAX_ERROR period={period} width={} sign={sign} ticks={max_error:.17} samples={}", $width, inputs.len());
                }
            }};
        }
        check!(Angle8, 8);
        check!(Angle16, 16);
        check!(Angle32, 32);
        check!(Angle64, 64);
        check!(Angle128, 128);
    }
}

#[test]
fn rounded_ratios_wrap_at_every_width() {
    macro_rules! check { ($($angle:ty),+) => {$({
        for (n,d) in [(1,1),(-1,1),(1,-1),(-1,-1)] { assert_eq!(<$angle>::from_turn_ratio(n,d), <$angle>::ZERO); }
        assert_eq!(<$angle>::from_turn_ratio(-1, i64::MAX), -<$angle>::from_turn_ratio(1, i64::MAX));
        assert_eq!(<$angle>::from_turn_ratio(i64::MAX - 1, i64::MAX), -<$angle>::from_turn_ratio(1, i64::MAX));
    })+}; }
    check!(Angle8, Angle16, Angle32, Angle64, Angle128);
}
