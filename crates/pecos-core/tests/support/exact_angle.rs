// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

use num_traits::ToPrimitive;
use pecos_core::BitUInt;

// Independent arbitrary-width rational oracle using BitUInt's general arithmetic.
// Production uses two carried 64-bit digits; this builds the entire numerator
// and denominator, divides, and checks the exact rounding error as integers.
pub fn expected(value: f64, period: f64, width: u16) -> (u128, f64) {
    let remainder = value % period;
    let parts = |x: f64| {
        let bits = x.to_bits();
        let exponent = i32::try_from((bits >> 52) & 0x7ff).unwrap();
        let mantissa = bits & ((1_u64 << 52) - 1);
        if exponent == 0 {
            (mantissa, -1074)
        } else {
            (mantissa | (1_u64 << 52), exponent - 1075)
        }
    };
    let (n, ne) = parts(remainder.abs());
    let (d, de) = parts(period);
    let shift = ne - de + i32::from(width);
    let capacity = u16::try_from(shift.abs() + 192).unwrap();
    let numerator = BitUInt::new(capacity, n) << u16::try_from(shift.max(0)).unwrap();
    let denominator = BitUInt::new(capacity, d) << u16::try_from((-shift).max(0)).unwrap();
    let one = BitUInt::new(capacity, 1);
    let quotient = &numerator / &denominator;
    let residual = &numerator - &(&quotient * &denominator);
    let rounded = if (&residual << 1) >= denominator {
        &quotient + &one
    } else {
        quotient
    };
    let represented = &rounded * &denominator;
    let error = if represented >= numerator {
        &represented - &numerator
    } else {
        &numerator - &represented
    };
    assert!(
        (&error << 1) <= denominator,
        "oracle error exceeds half a tick"
    );
    // Convert only the error report to f64. All decisions above are exact.
    let top = (0..capacity)
        .rev()
        .find(|&i| denominator.get_bit(i))
        .unwrap();
    let trim = top.saturating_sub(52);
    let report = (&error >> trim).to_u64().unwrap().to_f64().unwrap()
        / (&denominator >> trim).to_u64().unwrap().to_f64().unwrap();
    let words = rounded.to_words();
    let magnitude = u128::from(words[0]) | (u128::from(words[1]) << 64);
    let signed = if remainder.is_sign_negative() {
        magnitude.wrapping_neg()
    } else {
        magnitude
    };
    (signed & (u128::MAX >> (128 - width)), report)
}
