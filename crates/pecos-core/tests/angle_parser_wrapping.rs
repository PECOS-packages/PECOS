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
