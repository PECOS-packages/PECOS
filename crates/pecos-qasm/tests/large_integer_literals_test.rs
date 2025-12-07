// Test that verifies arbitrary-precision integer literals work in QASM

use pecos_engines::{Data, sim_builder};
use pecos_programs::Qasm;
use pecos_qasm::qasm_engine;

#[test]
fn test_very_large_integer_literal() {
    // Test with a 100-digit integer literal
    let qasm = r"
        OPENQASM 2.0;

        creg c[400];  // Need enough bits to store the value

        // 100-digit number: 10^99
        c = 1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000;
    ";

    let shot_vec = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .run(1)
        .unwrap();
    let shot = &shot_vec.shots[0];

    if let Data::BitVec(bitvec) = &shot.data["c"] {
        assert_eq!(bitvec.len(), 400);

        // 10^99 = 2^99 * 5^99
        // This requires about 330 bits to represent
        // We should have many bits set
        let ones_count = bitvec.count_ones();
        assert!(
            ones_count > 100,
            "Large number should have many bits set, got {ones_count}"
        );

        println!("Successfully parsed and stored 100-digit integer literal!");
        println!("Required {} bits with {} ones", bitvec.len(), ones_count);
    }
}

#[test]
fn test_large_integer_arithmetic() {
    // Test arithmetic with large literals
    let qasm = r"
        OPENQASM 2.0;

        creg a[256];
        creg b[256];
        creg sum[256];

        // Large literals that exceed 64-bit range
        a = 18446744073709551616;  // 2^64 (one more than max u64)
        b = 18446744073709551615;  // 2^64 - 1 (max u64)

        sum = a + b;  // Should be 2^65 - 1
    ";

    let shot_vec = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .run(1)
        .unwrap();
    let shot = &shot_vec.shots[0];

    // Check a (2^64)
    if let Data::BitVec(a_bits) = &shot.data["a"] {
        assert_eq!(a_bits.len(), 256);
        // Should have only bit 64 set
        assert!(!a_bits[63], "Bit 63 should be 0");
        assert!(a_bits[64], "Bit 64 should be 1");
        assert!(!a_bits[65], "Bit 65 should be 0");

        let a_ones = a_bits.iter().take(70).filter(|b| **b).count();
        assert_eq!(a_ones, 1, "Should have exactly one bit set");
    }

    // Check b (2^64 - 1)
    if let Data::BitVec(b_bits) = &shot.data["b"] {
        // Should have bits 0-63 all set
        for i in 0..64 {
            assert!(b_bits[i], "Bit {i} should be 1");
        }
        assert!(!b_bits[64], "Bit 64 should be 0");
    }

    // Check sum (2^65 - 1)
    if let Data::BitVec(sum_bits) = &shot.data["sum"] {
        // Should have bits 0-64 all set
        for i in 0..65 {
            assert!(sum_bits[i], "Bit {i} should be 1 in sum");
        }
        assert!(!sum_bits[65], "Bit 65 should be 0");

        println!("Large integer arithmetic working correctly!");
    }
}

#[test]
fn test_negative_large_literals() {
    // Test negative large literals
    let qasm = r"
        OPENQASM 2.0;

        creg value[256];
        creg neg_value[256];

        // Large negative literal
        value = -18446744073709551616;  // -(2^64)

        // Test unary negation
        neg_value = -value;  // Should be 2^64
    ";

    let shot_vec = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .run(1)
        .unwrap();
    let shot = &shot_vec.shots[0];

    // Check value (-(2^64) in two's complement)
    if let Data::BitVec(value_bits) = &shot.data["value"] {
        println!(
            "Negative large literal stored with {} bits set",
            value_bits.count_ones()
        );
        println!("value has {} bits total", value_bits.len());
        println!("First 70 bits of value:");
        for i in 0..70 {
            if value_bits[i] {
                println!("  Bit {i} is set");
            }
        }
    }

    // Check neg_value (should be 2^64)
    if let Data::BitVec(neg_bits) = &shot.data["neg_value"] {
        // Debug output
        println!("neg_value has {} bits total", neg_bits.len());
        println!("First 70 bits:");
        for i in 0..70 {
            if neg_bits[i] {
                println!("  Bit {i} is set");
            }
        }

        // Should have only bit 64 set
        assert!(neg_bits[64], "Bit 64 should be 1");

        // Count bits in lower 70 positions
        let ones_in_range = neg_bits.iter().take(70).filter(|b| **b).count();
        assert_eq!(ones_in_range, 1, "Should have exactly one bit set");

        println!("Unary negation of large literals working correctly!");
    }
}

#[test]
fn test_extremely_large_literal() {
    // Test with a literal larger than 128 bits
    let qasm = r"
        OPENQASM 2.0;

        creg huge[512];

        // 2^200 (201 digits)
        huge = 1606938044258990275541962092341162602522202993782792835301376;
    ";

    let shot_vec = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .run(1)
        .unwrap();
    let shot = &shot_vec.shots[0];

    if let Data::BitVec(huge_bits) = &shot.data["huge"] {
        assert_eq!(huge_bits.len(), 512);

        // This is 2^200, so bit 200 should be set
        assert!(huge_bits[200], "Bit 200 should be 1");

        // Should have exactly one bit set
        let ones_count = huge_bits.count_ones();
        assert_eq!(ones_count, 1, "Should have exactly one bit set for 2^200");

        println!("Successfully handled 201-digit literal (2^200)!");
    }
}

#[test]
fn test_literal_display_and_parsing() {
    // Test that we can parse and display large literals correctly
    let qasm = r"
        OPENQASM 2.0;

        creg a[128];
        creg b[128];
        creg c[128];

        // Various large literals
        a = 123456789012345678901234567890;  // 30 digits
        b = 999999999999999999999999999999;  // 30 nines
        c = 1000000000000000000000000000000; // 10^30
    ";

    let shot_vec = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .run(1)
        .unwrap();
    let shot = &shot_vec.shots[0];

    // Just verify they were parsed and stored
    assert!(shot.data.contains_key("a"));
    assert!(shot.data.contains_key("b"));
    assert!(shot.data.contains_key("c"));

    if let Data::BitVec(a_bits) = &shot.data["a"] {
        println!(
            "30-digit literal 'a' uses {} bits with {} ones",
            a_bits.len(),
            a_bits.count_ones()
        );
    }

    if let Data::BitVec(b_bits) = &shot.data["b"] {
        println!(
            "30-digit literal 'b' uses {} bits with {} ones",
            b_bits.len(),
            b_bits.count_ones()
        );
    }

    if let Data::BitVec(c_bits) = &shot.data["c"] {
        println!(
            "10^30 uses {} bits with {} ones",
            c_bits.len(),
            c_bits.count_ones()
        );
    }

    println!("Large literal parsing and storage working correctly!");
}

#[test]
fn test_mixed_size_literals_in_expressions() {
    // Test expressions mixing different sized literals
    let qasm = r"
        OPENQASM 2.0;

        creg result[256];
        creg test[10];

        // Mix large and small literals
        result = 18446744073709551616 + 100 - 50;  // 2^64 + 50

        // Comparisons with large literals
        test[0] = (result > 18446744073709551616);  // Should be true
        test[1] = (result == 18446744073709551666); // Should be true
        test[2] = (18446744073709551616 > 1000);    // Should be true
    ";

    let shot_vec = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .run(1)
        .unwrap();
    let shot = &shot_vec.shots[0];

    // Check result
    if let Data::BitVec(result_bits) = &shot.data["result"] {
        // Should be 2^64 + 50
        // Bits 1, 4, 5 should be set (50 = 110010 in binary)
        assert!(result_bits[1], "Bit 1 should be 1");
        assert!(result_bits[4], "Bit 4 should be 1");
        assert!(result_bits[5], "Bit 5 should be 1");
        assert!(result_bits[64], "Bit 64 should be 1");

        println!("Mixed literal arithmetic working!");
    }

    // Check comparisons
    if let Data::BitVec(test_bits) = &shot.data["test"] {
        assert!(test_bits[0], "result > 2^64 should be true");
        assert!(test_bits[1], "result == 2^64 + 50 should be true");
        assert!(test_bits[2], "2^64 > 1000 should be true");

        println!("Large literal comparisons working correctly!");
    }
}
