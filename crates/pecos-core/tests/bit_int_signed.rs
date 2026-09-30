use pecos_core::BitInt;

// Independent i128 oracle: never use BitInt equality or conversion to check a result.
fn from_i128(size: u16, value: i128) -> BitInt {
    let bytes = value.to_le_bytes();
    let mut words = bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|&chunk| u64::from_le_bytes(chunk))
        .collect::<Vec<_>>();
    words.resize(
        usize::from(size + 1).div_ceil(64),
        if value < 0 { u64::MAX } else { 0 },
    );
    BitInt::new_from_raw_inner(size, words.into_boxed_slice())
}

fn assert_value(actual: &BitInt, size: u16, expected: i128) {
    assert_eq!(actual.size(), size);
    assert_eq!(actual.inner().size(), size + 1);
    assert_eq!(
        actual.inner_words(),
        from_i128(size, expected).inner_words()
    );
}

fn check_arithmetic(size: u16, a: i128, b: i128) {
    let lhs = from_i128(size, a);
    let rhs = from_i128(size, b);
    for (actual, expected) in [
        (&lhs + &rhs, a + b),
        (&lhs - &rhs, a - b),
        (&lhs * &rhs, a * b),
        (&lhs & &rhs, a & b),
        (&lhs | &rhs, a | b),
        (&lhs ^ &rhs, a ^ b),
    ] {
        assert_value(&actual, size, expected);
    }
    if b != 0 {
        assert_value(&(&lhs / &rhs), size, a / b);
        assert_value(&(&lhs % &rhs), size, a % b);
    }
}

#[test]
fn equal_width_compatibility_through_62() {
    for size in 1..=5 {
        let limit = 1i128 << size;
        for a in -limit..limit {
            for b in -limit..limit {
                check_arithmetic(size, a, b);
            }
        }
    }
    for size in 1..=62 {
        let limit = 1i128 << size;
        let values = [
            -limit,
            -limit + 1,
            -17,
            -2,
            -1,
            0,
            1,
            2,
            17,
            limit - 2,
            limit - 1,
        ];
        for a in values.into_iter().filter(|&v| (-limit..limit).contains(&v)) {
            for b in values.into_iter().filter(|&v| (-limit..limit).contains(&v)) {
                check_arithmetic(size, a, b);
            }
        }
    }
}

#[test]
fn wide_division_packet_anchors() {
    for (size, a, b, quotient, remainder) in [
        (64, -8, 2, -4, 0),
        (64, -7, 2, -3, -1),
        (64, 7, -2, -3, 1),
        (100, -(1i128 << 70), 2, -590_295_810_358_705_651_712, 0),
        (100, 1i128 << 70, -2, -590_295_810_358_705_651_712, 0),
        (
            100,
            -((1i128 << 70) + 1),
            3,
            -393_530_540_239_137_101_141,
            -2,
        ),
        (63, i128::from(i64::MIN), -1, i128::from(i64::MIN), 0),
        // Zero / nonzero is zero; a smaller magnitude gives quotient zero
        // and leaves the entire dividend as the remainder.
        (100, 0, -(1i128 << 70), 0, 0),
        (100, -7, 1i128 << 70, 0, -7),
        (100, 7, -(1i128 << 70), 0, 7),
    ] {
        let lhs = from_i128(size, a);
        let rhs = from_i128(size, b);
        assert_value(&(&lhs / &rhs), size, quotient);
        assert_value(&(&lhs % &rhs), size, remainder);
    }
}

#[test]
fn minimum_patterns_and_wrapping_division() {
    for (size, minimum, pattern) in [
        (1, -2, 2u128),
        (8, -256, 256),
        (64, -18_446_744_073_709_551_616, 18_446_744_073_709_551_616),
        (
            100,
            -1_267_650_600_228_229_401_496_703_205_376,
            1_267_650_600_228_229_401_496_703_205_376,
        ),
    ] {
        let lhs = from_i128(size, minimum);
        let words = lhs.inner_words();
        let actual = u128::from(words[0]) | (u128::from(words.get(1).copied().unwrap_or(0)) << 64);
        assert_eq!(actual, pattern);
        assert_value(&(&lhs / BitInt::new(size, -1)), size, minimum);
        assert_value(&(&lhs % BitInt::new(size, -1)), size, 0);
    }
}

#[test]
fn wide_equality_uses_all_words() {
    let zero = BitInt::zero(100);
    let high = from_i128(100, 1i128 << 70);
    assert_ne!(zero, high);
    assert_ne!(
        from_i128(100, -(1i128 << 70)),
        from_i128(100, -(1i128 << 69))
    );
}

#[test]
fn wide_ordering_uses_all_words() {
    for (size, a, b) in [
        (100, 0, 1i128 << 70),
        (100, 1i128 << 70, 1i128 << 71),
        (100, -(1i128 << 70), -(1i128 << 69)),
        (70, 1, 1i128 << 65),
    ] {
        let lhs = from_i128(size, a);
        let rhs = from_i128(size, b);
        assert!(lhs < rhs);
        assert!(rhs > lhs);
        assert_eq!(lhs.partial_cmp(&rhs), Some(std::cmp::Ordering::Less));
    }
}

#[test]
fn cross_width_comparison_by_value() {
    for (left_size, a, right_size, b, equal, less) in [
        (1, -1, 2, -1, true, false),
        (1, -1, 2, 3, false, true),
        (1, -1, 2, -2, false, false),
        (8, -1, 100, -1, true, false),
        (100, 1i128 << 70, 130, 1i128 << 70, true, false),
        (100, -(1i128 << 70), 130, -(1i128 << 70), true, false),
    ] {
        let lhs = from_i128(left_size, a);
        let rhs = from_i128(right_size, b);
        assert_eq!(lhs == rhs, equal);
        assert_eq!(rhs == lhs, equal);
        assert_eq!(lhs < rhs, less);
        assert_eq!(lhs.cmp(&rhs), a.cmp(&b));
        assert_eq!(rhs.cmp(&lhs), b.cmp(&a));
    }
}

#[test]
fn mixed_width_packet_anchors() {
    for (left_size, a, right_size, b, expected) in [
        (1, -2, 2, -1, [-2, 0, 1, -1, -2]),
        // Declared width 2 cannot hold positive 4: it spans -4 through 3.
        (1, 1, 3, 4, [0, 1, 1, 1, 0]),
        (2, 0, 1, -1, [0, 0, -1, 1, 0]),
        (2, 1, 1, -1, [-1, 0, 0, 2, -1]),
    ] {
        let lhs = BitInt::new(left_size, a);
        let rhs = BitInt::new(right_size, b);
        for (actual, value) in [
            &lhs / &rhs,
            &lhs % &rhs,
            &lhs + &rhs,
            &lhs - &rhs,
            &lhs * &rhs,
        ]
        .iter()
        .zip(expected)
        {
            assert_value(actual, left_size, value);
        }
    }
}

#[test]
fn binary_operators_match_independent_mixed_width_oracle() {
    for left_size in 1..=4 {
        for right_size in 1..=4 {
            for a in -(1i128 << left_size)..(1i128 << left_size) {
                for b in -(1i128 << right_size)..(1i128 << right_size) {
                    let lhs = from_i128(left_size, a);
                    let rhs = from_i128(right_size, b);
                    assert_eq!(lhs.cmp(&rhs), a.cmp(&b));
                    for (actual, expected) in [
                        (&lhs + &rhs, a + b),
                        (&lhs - &rhs, a - b),
                        (&lhs * &rhs, a * b),
                        (&lhs & &rhs, a & b),
                        (&lhs | &rhs, a | b),
                        (&lhs ^ &rhs, a ^ b),
                    ] {
                        assert_value(&actual, left_size, expected);
                    }
                    if b != 0 {
                        assert_value(&(&lhs / &rhs), left_size, a / b);
                        assert_value(&(&lhs % &rhs), left_size, a % b);
                    }
                }
            }
        }
    }
    // Cross storage boundaries in both directions, including partial top words.
    for left_size in [8, 63, 64, 65, 100, 130] {
        for right_size in [8, 63, 64, 65, 100, 130] {
            let a = if left_size > 70 {
                -(1i128 << 70) + 1
            } else {
                -7
            };
            let b = if right_size > 70 {
                (1i128 << 70) + 3
            } else {
                -3
            };
            let lhs = from_i128(left_size, a);
            let rhs = from_i128(right_size, b);
            // Products here use a small factor so the independent i128 oracle fits.
            for (actual, expected) in [
                (&lhs + &rhs, a + b),
                (&lhs - &rhs, a - b),
                (&lhs & &rhs, a & b),
                (&lhs | &rhs, a | b),
                (&lhs ^ &rhs, a ^ b),
                (&lhs / &rhs, a / b),
                (&lhs % &rhs, a % b),
                (&lhs * BitInt::new(right_size, -3), a * -3),
            ] {
                assert_value(&actual, left_size, expected);
            }
        }
    }
}

#[test]
fn wide_i64_conversion_checks_value() {
    for size in [63, 64, 65, 100, 130, 65534] {
        for value in [i64::MIN, i64::MIN + 1, -1, 0, 1, i64::MAX] {
            assert_eq!(BitInt::new(size, value).to_i64(), Some(value));
        }
    }
    // One beyond either endpoint cannot be represented as i64.
    for size in [64, 100, 130] {
        assert_eq!(from_i128(size, i128::from(i64::MAX) + 1).to_i64(), None);
        assert_eq!(from_i128(size, i128::from(i64::MIN) - 1).to_i64(), None);
    }
}

#[test]
#[should_panic(expected = "Division by zero")]
fn wide_division_by_zero_message() {
    let _ = from_i128(100, 1i128 << 70) / BitInt::zero(130);
}

#[test]
#[should_panic(expected = "Remainder by zero")]
fn wide_remainder_by_zero_message() {
    let _ = from_i128(100, 1i128 << 70) % BitInt::zero(130);
}

#[test]
fn binary_operators_random_i128_oracle() {
    use rand::{RngExt, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(876);
    for left_size in [1, 8, 33, 62, 63, 64, 65, 100, 126] {
        for right_size in [1, 8, 33, 62, 63, 64, 65, 100, 126] {
            for _ in 0..100 {
                let a = rng.random::<i128>() >> (127 - left_size);
                let b = rng.random::<i128>() >> (127 - right_size);
                let lhs = from_i128(left_size, a);
                let rhs = from_i128(right_size, b);
                assert_eq!(lhs == rhs, a == b);
                assert_eq!(lhs.cmp(&rhs), a.cmp(&b));
                assert_eq!(lhs.to_i64(), i64::try_from(a).ok());
                for (actual, expected) in [
                    (&lhs + &rhs, a.wrapping_add(b)),
                    (&lhs - &rhs, a.wrapping_sub(b)),
                    (&lhs * &rhs, a.wrapping_mul(b)),
                    (&lhs & &rhs, a & b),
                    (&lhs | &rhs, a | b),
                    (&lhs ^ &rhs, a ^ b),
                ] {
                    // i128 wrapping preserves every bit needed by widths <= 127.
                    assert_value(&actual, left_size, expected);
                }
                if b != 0 {
                    assert_value(&(&lhs / &rhs), left_size, a / b);
                    assert_value(&(&lhs % &rhs), left_size, a % b);
                }
            }
        }
    }
}
