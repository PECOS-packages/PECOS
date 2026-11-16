// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

use pecos_core::{Angle64, Angle128, ParseAngleError};

#[test]
fn pi_fractions_wrap_before_ratio_conversion() {
    // Integer fractions of 2^64 / 2^128, independent of the ratio constructor.
    for (text, quarters) in [
        ("3pi", 2_u128),
        ("5pi/2", 1),
        ("-7pi/2", 1),
        ("2pi", 0),
        ("5pi/-2", 3),
        ("9223372036854775807pi", 2),
        ("-9223372036854775808pi", 0),
    ] {
        assert_eq!(
            Angle64::from_str_radians(text).unwrap().fraction(),
            u64::try_from(quarters << 62).unwrap()
        );
        assert_eq!(
            Angle128::from_str_radians(text).unwrap().fraction(),
            quarters << 126
        );
    }
}

#[test]
fn decimal_overflow_has_decimal_remedy() {
    for suffix in ["", "pi"] {
        let text = format!("{}.0{suffix}", "9".repeat(309));
        for error in [
            Angle64::from_str_radians(&text).unwrap_err(),
            Angle128::from_str_radians(&text).unwrap_err(),
        ] {
            assert!(matches!(error, ParseAngleError::DecimalOverflow { .. }));
            let message = error.to_string();
            assert!(message.contains(&text));
            assert!(!message.contains("numerator"));
        }
    }
}

#[test]
fn pi_fractions_accept_every_i64_denominator() {
    // Exact ticks: 2^bits/(2*d), rounded once. For d=2^63-1,
    // 2^127/d = 2^64+2 + 2/(2^63-1), so the 128-bit result is 2^64+2.
    for (denominator, expected64, expected128) in [
        (1_i64 << 62, 2_u64, 1_u128 << 65),
        (i64::MAX, 1, (1_u128 << 64) + 2),
        (i64::MIN, u64::MAX, (1_u128 << 64).wrapping_neg()),
    ] {
        for (numerator, negative) in [("", false), ("-", true)] {
            let text = format!("{numerator}pi/{denominator}");
            assert_eq!(
                Angle64::from_str_radians(&text).unwrap().fraction(),
                if negative {
                    expected64.wrapping_neg()
                } else {
                    expected64
                }
            );
            assert_eq!(
                Angle128::from_str_radians(&text).unwrap().fraction(),
                if negative {
                    expected128.wrapping_neg()
                } else {
                    expected128
                }
            );
        }
    }
}
