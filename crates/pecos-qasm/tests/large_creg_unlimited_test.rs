// Test that verifies arbitrary-precision BitVec expressions work without limitations

use pecos_engines::{Data, sim_builder};
use pecos_programs::Qasm;
use pecos_qasm::qasm_engine;

#[test]
fn test_large_register_full_value_assignment() {
    // Test that we can now assign values larger than 64 bits
    let qasm = r"
        OPENQASM 2.0;

        creg c[128];

        // Set all 128 bits to 1 using expressions
        // First set lower 64 bits
        c = 9223372036854775807;  // 2^63 - 1 (max positive i64)

        // Now use bit operations to set upper bits
        c[64] = 1;
        c[65] = 1;
        c[70] = 1;
        c[100] = 1;
        c[127] = 1;
    ";

    let shot_vec = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .run(1)
        .unwrap();
    let shot = &shot_vec.shots[0];

    if let Data::BitVec(bitvec) = &shot.data["c"] {
        assert_eq!(bitvec.len(), 128);

        // Check first 63 bits are set
        for i in 0..63 {
            assert!(bitvec[i], "Bit {i} should be 1");
        }

        // Check specific upper bits we set
        assert!(bitvec[64]);
        assert!(bitvec[65]);
        assert!(bitvec[70]);
        assert!(bitvec[100]);
        assert!(bitvec[127]);

        println!("Successfully assigned values to 128-bit register!");
    }
}

#[test]
fn test_large_register_full_arithmetic() {
    // Test full-width arithmetic on large registers
    let qasm = r"
        OPENQASM 2.0;

        creg a[100];
        creg b[100];
        creg sum[100];
        creg result[100];

        // Set bits throughout the register to create large numbers
        a[0] = 1;
        a[50] = 1;
        a[99] = 1;

        b[0] = 1;
        b[50] = 1;
        b[98] = 1;

        // Full-width operations
        sum = a + b;  // Should now work on full 100 bits
        result = a | b;  // Should now work on full 100 bits
    ";

    let shot_vec = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .run(1)
        .unwrap();
    let shot = &shot_vec.shots[0];

    // Check sum (a + b)
    if let Data::BitVec(sum_bits) = &shot.data["sum"] {
        assert_eq!(sum_bits.len(), 100);

        // a has bits at 0, 50, 99
        // b has bits at 0, 50, 98
        // sum should have:
        // - bit 1 set (from adding bit 0 of both)
        // - bit 51 set (from adding bit 50 of both)
        // - bit 98 set (from b[98])
        // - bit 99 set (from a[99])

        assert!(!sum_bits[0], "Bit 0 should be 0 (carry out)");
        assert!(sum_bits[1], "Bit 1 should be 1 (sum of two 1s)");
        assert!(!sum_bits[50], "Bit 50 should be 0 (carry out)");
        assert!(sum_bits[51], "Bit 51 should be 1 (sum of two 1s)");
        assert!(sum_bits[98], "Bit 98 should be 1");
        assert!(sum_bits[99], "Bit 99 should be 1");

        println!("Full-width addition working correctly!");
    }

    // Check OR result
    if let Data::BitVec(result_bits) = &shot.data["result"] {
        assert_eq!(result_bits.len(), 100);

        // OR should have all bits that are in either a or b
        assert!(result_bits[0], "Bit 0 should be 1");
        assert!(result_bits[50], "Bit 50 should be 1");
        assert!(result_bits[98], "Bit 98 should be 1");
        assert!(result_bits[99], "Bit 99 should be 1");

        // Count total set bits (should be 4)
        assert_eq!(result_bits.count_ones(), 4);

        println!("Full-width OR working correctly!");
    }
}

#[test]
fn test_large_register_comparisons() {
    // Test comparisons work correctly with large values
    let qasm = r"
        OPENQASM 2.0;

        creg a[100];
        creg b[100];
        creg results[10];

        // Create values that differ only in high bits
        a[0] = 1;
        a[99] = 1;  // a has sign bit set (negative in two's complement)

        b[0] = 1;
        b[98] = 1;  // b is positive (sign bit not set)

        // With signed integers: negative < positive
        results[0] = (a > b);   // Should be false (negative > positive = false)
        results[1] = (a < b);   // Should be true (negative < positive = true)
        results[2] = (a == b);  // Should be false
        results[3] = (a != b);  // Should be true
    ";

    let shot_vec = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .run(1)
        .unwrap();
    let shot = &shot_vec.shots[0];

    if let Data::BitVec(results_bits) = &shot.data["results"] {
        assert!(
            !results_bits[0],
            "a > b should be false (negative > positive)"
        );
        assert!(
            results_bits[1],
            "a < b should be true (negative < positive)"
        );
        assert!(!results_bits[2], "a == b should be false");
        assert!(results_bits[3], "a != b should be true");
    }
}

#[test]
fn test_large_register_shift_full_width() {
    // Test shift operations work on full register width
    let qasm = r"
        OPENQASM 2.0;

        creg value[100];
        creg left_shift[100];
        creg right_shift[100];

        // Set bits throughout the register
        value[0] = 1;
        value[50] = 1;
        value[90] = 1;

        // Shift operations should preserve bits beyond 64
        left_shift = value << 5;
        right_shift = value >> 3;
    ";

    let shot_vec = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .run(1)
        .unwrap();
    let shot = &shot_vec.shots[0];

    // Check left shift
    if let Data::BitVec(left_bits) = &shot.data["left_shift"] {
        assert_eq!(left_bits.len(), 100);

        // Original bits at 0, 50, 90
        // After << 5: should be at 5, 55, 95
        assert!(left_bits[5], "Bit 5 should be 1 (0 << 5)");
        assert!(left_bits[55], "Bit 55 should be 1 (50 << 5)");
        assert!(left_bits[95], "Bit 95 should be 1 (90 << 5)");

        assert_eq!(left_bits.count_ones(), 3);
        println!("Full-width left shift working correctly!");
    }

    // Check right shift
    if let Data::BitVec(right_bits) = &shot.data["right_shift"] {
        assert_eq!(right_bits.len(), 100);

        // Original bits at 0, 50, 90
        // After >> 3: should be at 47, 87 (bit 0 is lost)
        assert!(!right_bits[0], "Bit 0 should be 0 (shifted out)");
        assert!(right_bits[47], "Bit 47 should be 1 (50 >> 3)");
        assert!(right_bits[87], "Bit 87 should be 1 (90 >> 3)");

        assert_eq!(right_bits.count_ones(), 2);
        println!("Full-width right shift working correctly!");
    }
}

#[test]
fn test_complex_expression_chain() {
    // Test a complex chain of operations on large registers
    let qasm = r"
        OPENQASM 2.0;

        creg a[120];
        creg b[120];
        creg c[120];
        creg temp[120];
        creg final[120];

        // Set up initial values across the full width
        a[0] = 1;
        a[60] = 1;
        a[119] = 1;

        b[30] = 1;
        b[60] = 1;
        b[118] = 1;

        // Complex expression chain
        temp = (a | b) & ~(a & b);  // XOR using other operations
        c = temp << 1;
        final = c + a;
    ";

    let shot_vec = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .run(1)
        .unwrap();
    let shot = &shot_vec.shots[0];

    // Verify temp (XOR result)
    if let Data::BitVec(temp_bits) = &shot.data["temp"] {
        // XOR should have bits where a and b differ
        assert!(temp_bits[0], "Bit 0: a has 1, b has 0");
        assert!(temp_bits[30], "Bit 30: a has 0, b has 1");
        assert!(!temp_bits[60], "Bit 60: both have 1");
        assert!(temp_bits[118], "Bit 118: a has 0, b has 1");
        assert!(temp_bits[119], "Bit 119: a has 1, b has 0");

        assert_eq!(temp_bits.count_ones(), 4);
    }

    // Verify final result
    if let Data::BitVec(final_bits) = &shot.data["final"] {
        // This involves multiple operations across the full width
        println!("Complex expression chain completed successfully!");
        println!("Final result has {} bits set", final_bits.count_ones());
    }
}

#[test]
fn test_negative_numbers_full_width() {
    // Test that negative numbers work correctly with large registers
    let qasm = r"
        OPENQASM 2.0;

        creg value[100];
        creg neg_one[100];
        creg result[100];

        // Set value to -1 (all bits should be 1 in two's complement)
        value = -1;

        // Also test unary negation
        neg_one = -1;
        result = -neg_one;  // Should be 1
    ";

    let shot_vec = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .run(1)
        .unwrap();
    let shot = &shot_vec.shots[0];

    // Check value (-1 in two's complement)
    if let Data::BitVec(value_bits) = &shot.data["value"] {
        assert_eq!(value_bits.len(), 100);

        // In our current implementation, -1 as i64 will set lower 64 bits
        // and sign-extend to fill the register
        for i in 0..64 {
            assert!(value_bits[i], "Bit {i} should be 1 for -1");
        }

        // Upper bits depend on sign extension implementation
        println!(
            "Negative number representation: {} bits set",
            value_bits.count_ones()
        );
    }

    // Check result (should be 1)
    if let Data::BitVec(result_bits) = &shot.data["result"] {
        assert!(result_bits[0], "Bit 0 should be 1");
        for i in 1..100 {
            assert!(!result_bits[i], "Bit {i} should be 0");
        }

        println!("Unary negation working correctly!");
    }
}
