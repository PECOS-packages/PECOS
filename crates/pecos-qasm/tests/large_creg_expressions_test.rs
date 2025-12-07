use pecos_engines::{Data, sim_builder};
use pecos_programs::Qasm;
use pecos_qasm::qasm_engine;

#[test]
fn test_large_creg_bitwise_expressions() {
    // Test bitwise operations with large classical registers
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        creg a[80];
        creg b[80];
        creg c[80];

        // Set bits in register a
        a[0] = 1;
        a[1] = 1;
        a[63] = 1;
        a[64] = 1;
        a[79] = 1;

        // Set bits in register b
        b[0] = 1;
        b[2] = 1;
        b[63] = 1;
        b[65] = 1;
        b[79] = 1;

        // Test conditionals with large registers
        if (a[64] == 1) c[0] = 1;
        if (b[65] == 1) c[1] = 1;
        if (a[79] == 1) c[2] = 1;

        // Test bit access in expressions
        c[10] = a[0] & b[0];  // Should be 1
        c[11] = a[1] & b[1];  // Should be 0
        c[12] = a[64] & b[64]; // Should be 0
        c[13] = a[79] & b[79]; // Should be 1
    "#;

    let shot_vec = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .run(1)
        .unwrap();
    let shot = &shot_vec.shots[0];

    // Check register a
    if let Data::BitVec(bitvec_a) = &shot.data["a"] {
        assert_eq!(bitvec_a.len(), 80);
        assert!(bitvec_a[0]);
        assert!(bitvec_a[1]);
        assert!(bitvec_a[63]);
        assert!(bitvec_a[64]);
        assert!(bitvec_a[79]);
        assert_eq!(bitvec_a.count_ones(), 5);
    }

    // Check register b
    if let Data::BitVec(bitvec_b) = &shot.data["b"] {
        assert_eq!(bitvec_b.len(), 80);
        assert!(bitvec_b[0]);
        assert!(bitvec_b[2]);
        assert!(bitvec_b[63]);
        assert!(bitvec_b[65]);
        assert!(bitvec_b[79]);
        assert_eq!(bitvec_b.count_ones(), 5);
    }

    // Check register c (results)
    if let Data::BitVec(bitvec_c) = &shot.data["c"] {
        assert_eq!(bitvec_c.len(), 80);

        // Check conditional results
        assert!(bitvec_c[0], "c[0] should be 1 (a[64] == 1)");
        assert!(bitvec_c[1], "c[1] should be 1 (b[65] == 1)");
        assert!(bitvec_c[2], "c[2] should be 1 (a[79] == 1)");

        // Check bitwise AND results
        assert!(bitvec_c[10], "c[10] should be 1 (a[0] & b[0])");
        assert!(!bitvec_c[11], "c[11] should be 0 (a[1] & b[1])");
        assert!(!bitvec_c[12], "c[12] should be 0 (a[64] & b[64])");
        assert!(bitvec_c[13], "c[13] should be 1 (a[79] & b[79])");

        println!("Successfully tested bitwise expressions with large registers!");
    }
}

#[test]
fn test_large_creg_in_quantum_conditionals() {
    // Test using large classical registers to control quantum gates
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[5];
        creg control[100];
        creg result[5];

        // Set control bits at various positions
        control[0] = 1;
        control[50] = 1;
        control[99] = 1;

        // Use bits beyond 64-bit boundary for quantum control
        if (control[0] == 1) x q[0];
        if (control[50] == 1) x q[1];
        if (control[99] == 1) x q[2];

        // For complex expressions, we need to compute into a temp register
        creg temp[2];
        temp[0] = control[0] & control[50];
        temp[1] = control[50] & control[99];

        if (temp[0] == 1) x q[3];
        if (temp[1] == 1) x q[4];

        measure q -> result;
    "#;

    let shot_vec = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .run(1)
        .unwrap();
    let shot = &shot_vec.shots[0];

    // Check control register
    if let Data::BitVec(control_bits) = &shot.data["control"] {
        assert_eq!(control_bits.len(), 100);
        assert!(control_bits[0]);
        assert!(control_bits[50]);
        assert!(control_bits[99]);
        assert_eq!(control_bits.count_ones(), 3);
    }

    // Check measurement results
    if let Data::BitVec(result_bits) = &shot.data["result"] {
        assert_eq!(result_bits.len(), 5);
        assert!(result_bits[0], "q[0] should be 1 (control[0] == 1)");
        assert!(result_bits[1], "q[1] should be 1 (control[50] == 1)");
        assert!(result_bits[2], "q[2] should be 1 (control[99] == 1)");
        assert!(
            result_bits[3],
            "q[3] should be 1 (control[0] & control[50])"
        );
        assert!(
            result_bits[4],
            "q[4] should be 1 (control[50] & control[99])"
        );

        println!("Successfully used large registers in quantum conditionals!");
    }
}

#[test]
fn test_large_creg_arithmetic_expressions() {
    // Test arithmetic expressions with large registers
    let qasm = r"
        OPENQASM 2.0;

        creg a[72];
        creg b[72];
        creg sum[72];
        creg result[72];

        // Set some values
        a[0] = 1;
        a[1] = 1;
        a[2] = 1;  // a has value 7 in lower bits

        b[0] = 1;
        b[2] = 1;  // b has value 5 in lower bits

        // These work on the lower 64 bits due to i64 limitation
        sum = a + b;  // Should be 12 in lower bits

        // Individual bit checks beyond 64-bit boundary
        a[70] = 1;
        b[70] = 1;

        // Test expressions with individual bits
        result[0] = (a[0] & b[0]);  // 1 & 1 = 1
        result[1] = (a[1] | b[1]);  // 1 | 0 = 1
        result[2] = (a[2] ^ b[2]);  // 1 ^ 1 = 0
        result[70] = (a[70] & b[70]); // 1 & 1 = 1
    ";

    let shot_vec = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .run(1)
        .unwrap();
    let shot = &shot_vec.shots[0];

    // Check sum register (should be 12 = 1100 in binary)
    if let Data::BitVec(sum_bits) = &shot.data["sum"] {
        assert_eq!(sum_bits.len(), 72);
        assert!(!sum_bits[0], "sum[0] should be 0");
        assert!(!sum_bits[1], "sum[1] should be 0");
        assert!(sum_bits[2], "sum[2] should be 1");
        assert!(sum_bits[3], "sum[3] should be 1");

        // Calculate the numeric value of lower bits
        let mut sum_value = 0u64;
        for i in 0..64 {
            if sum_bits[i] {
                sum_value |= 1 << i;
            }
        }
        assert_eq!(sum_value, 12, "Sum should be 12");
    }

    // Check result register
    if let Data::BitVec(result_bits) = &shot.data["result"] {
        assert_eq!(result_bits.len(), 72);
        assert!(result_bits[0], "result[0] should be 1 (1 & 1)");
        assert!(result_bits[1], "result[1] should be 1 (1 | 0)");
        assert!(!result_bits[2], "result[2] should be 0 (1 ^ 1)");
        assert!(result_bits[70], "result[70] should be 1 (1 & 1)");

        println!("Successfully tested arithmetic expressions with large registers!");
    }
}

#[test]
fn test_large_creg_comparison_expressions() {
    // Test comparison expressions with large registers
    let qasm = r"
        OPENQASM 2.0;

        creg a[90];
        creg b[90];
        creg results[10];

        // Set values
        a[0] = 1;
        a[1] = 1;  // a = 3 in lower bits

        b[0] = 1;
        b[2] = 1;  // b = 5 in lower bits

        // Comparisons work on the i64 representation
        results[0] = (a < b);   // 3 < 5 = true
        results[1] = (a > b);   // 3 > 5 = false
        results[2] = (a == b);  // 3 == 5 = false
        results[3] = (a != b);  // 3 != 5 = true

        // Test with bits beyond 64-bit boundary
        a[80] = 1;
        b[80] = 1;

        // Individual bit comparisons
        results[4] = (a[80] == 1);  // true
        results[5] = (b[80] == 1);  // true
        results[6] = (a[80] == b[80]); // true

        // Test edge cases
        results[7] = (a[89] == 0);  // true (unset bit)
        results[8] = (b[89] != 1);  // true (unset bit)
    ";

    let shot_vec = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .run(1)
        .unwrap();
    let shot = &shot_vec.shots[0];

    // Check results
    if let Data::BitVec(results_bits) = &shot.data["results"] {
        assert_eq!(results_bits.len(), 10);

        assert!(results_bits[0], "results[0]: 3 < 5 should be true");
        assert!(!results_bits[1], "results[1]: 3 > 5 should be false");
        assert!(!results_bits[2], "results[2]: 3 == 5 should be false");
        assert!(results_bits[3], "results[3]: 3 != 5 should be true");

        assert!(results_bits[4], "results[4]: a[80] == 1 should be true");
        assert!(results_bits[5], "results[5]: b[80] == 1 should be true");
        assert!(results_bits[6], "results[6]: a[80] == b[80] should be true");

        assert!(results_bits[7], "results[7]: a[89] == 0 should be true");
        assert!(results_bits[8], "results[8]: b[89] != 1 should be true");
    }
}

#[test]
fn test_large_creg_shift_operations() {
    // Test shift operations with large registers
    let qasm = r"
        OPENQASM 2.0;

        creg value[80];
        creg shifted_left[80];
        creg shifted_right[80];

        // Set a pattern
        value[0] = 1;
        value[1] = 1;
        value[2] = 1;  // value = 7 in lower bits

        // Shift operations (limited to 64-bit arithmetic)
        shifted_left = value << 2;   // Should be 28 (11100 in binary)
        shifted_right = value >> 1;  // Should be 3 (11 in binary)

        // Manual bit shifting beyond 64-bit boundary
        value[60] = 1;
        value[61] = 1;

        // We can still access and manipulate individual bits beyond 64
        shifted_left[62] = value[60];  // Manual left shift by 2
        shifted_left[63] = value[61];

        shifted_right[59] = value[60]; // Manual right shift by 1
        shifted_right[60] = value[61];
    ";

    let shot_vec = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .run(1)
        .unwrap();
    let shot = &shot_vec.shots[0];

    // Check shifted_left (should be 28 = 11100 in binary for lower bits)
    if let Data::BitVec(left_bits) = &shot.data["shifted_left"] {
        assert_eq!(left_bits.len(), 80);
        assert!(!left_bits[0]);
        assert!(!left_bits[1]);
        assert!(left_bits[2], "bit 2 should be 1");
        assert!(left_bits[3], "bit 3 should be 1");
        assert!(left_bits[4], "bit 4 should be 1");

        // Check manual shifts beyond 64-bit
        assert!(left_bits[62], "bit 62 should be 1 (manual shift)");
        assert!(left_bits[63], "bit 63 should be 1 (manual shift)");
    }

    // Check shifted_right (should be 3 = 11 in binary)
    if let Data::BitVec(right_bits) = &shot.data["shifted_right"] {
        assert_eq!(right_bits.len(), 80);
        assert!(right_bits[0], "bit 0 should be 1");
        assert!(right_bits[1], "bit 1 should be 1");
        assert!(!right_bits[2]);

        // Check manual shifts
        assert!(right_bits[59], "bit 59 should be 1 (manual shift)");
        assert!(right_bits[60], "bit 60 should be 1 (manual shift)");

        println!("Successfully tested shift operations with large registers!");
    }
}

#[test]
fn test_large_creg_complex_expressions() {
    // Test complex nested expressions
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[4];
        creg a[100];
        creg b[100];
        creg c[100];
        creg flags[10];

        // Set values across the register
        a[0] = 1;
        a[5] = 1;
        a[50] = 1;
        a[99] = 1;

        b[0] = 1;
        b[5] = 1;
        b[60] = 1;
        b[99] = 1;

        // Complex expressions with multiple operations
        flags[0] = (a[0] & b[0]) | (a[5] & b[5]);  // Should be 1
        flags[1] = (a[50] & b[50]) | (a[60] & b[60]); // Should be 0
        flags[2] = (a[99] == 1) & (b[99] == 1);  // Should be 1

        // Use intermediate computations for complex conditions
        creg temp[4];
        temp[0] = (a[0] & b[0]) & (a[99] & b[99]);
        temp[1] = (a[50] == 1) & (b[60] == 1);
        temp[2] = (a[99] == 1) & (b[99] == 1);

        if (temp[0] == 1) x q[0];  // Should execute
        if (temp[1] == 1) x q[1];  // Should execute
        if (temp[2] == 1) x q[2];  // Should execute (replaces nested if)

        // Complex arithmetic in conditions
        // Note: full register OR is limited to 64 bits, so we do it manually for bits beyond
        c = a | b;  // This will OR the lower 64 bits

        // Manually set bits beyond 64-bit boundary
        c[99] = a[99] | b[99];

        // Check if c has any bits set (we know it does)
        temp[3] = c[0] | c[5] | c[50] | c[60] | c[99];
        if (temp[3] == 1) x q[3];  // Should execute

        measure q[0] -> flags[6];
        measure q[1] -> flags[7];
        measure q[2] -> flags[8];
        measure q[3] -> flags[9];
    "#;

    let shot_vec = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .run(1)
        .unwrap();
    let shot = &shot_vec.shots[0];

    // Check flags
    if let Data::BitVec(flags_bits) = &shot.data["flags"] {
        assert_eq!(flags_bits.len(), 10);

        assert!(flags_bits[0], "flags[0] should be 1");
        assert!(!flags_bits[1], "flags[1] should be 0");
        assert!(flags_bits[2], "flags[2] should be 1");

        // Check quantum measurement results
        assert!(flags_bits[6], "q[0] should be 1");
        assert!(flags_bits[7], "q[1] should be 1");
        assert!(flags_bits[8], "q[2] should be 1");
        assert!(flags_bits[9], "q[3] should be 1");
    }

    // Check c register (should have all bits from a OR b)
    if let Data::BitVec(c_bits) = &shot.data["c"] {
        assert_eq!(c_bits.len(), 100);
        assert!(c_bits[0]);
        assert!(c_bits[5]);
        assert!(c_bits[50]);
        assert!(c_bits[60]);
        assert!(c_bits[99]);
        assert_eq!(c_bits.count_ones(), 5);

        println!("Successfully tested complex expressions with large registers!");
    }
}

#[test]
fn test_edge_cases_and_limitations() {
    // Test edge cases and current limitations
    let qasm = r"
        OPENQASM 2.0;

        creg huge[1000];  // 1000-bit register
        creg test[10];

        // Individual bit operations work at any position
        huge[0] = 1;
        huge[100] = 1;
        huge[500] = 1;
        huge[999] = 1;

        // Test boundary conditions
        test[0] = huge[0];     // Should be 1
        test[1] = huge[100];   // Should be 1
        test[2] = huge[500];   // Should be 1
        test[3] = huge[999];   // Should be 1

        // Test unset bits
        test[4] = huge[1];     // Should be 0
        test[5] = huge[998];   // Should be 0

        // Complex expression with very large register
        test[6] = (huge[0] & huge[999]);  // Should be 1
        test[7] = (huge[100] | huge[200]); // Should be 1
        test[8] = (huge[500] ^ huge[600]); // Should be 1 (1 XOR 0)
    ";

    let shot_vec = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .run(1)
        .unwrap();
    let shot = &shot_vec.shots[0];

    // Check huge register
    if let Data::BitVec(huge_bits) = &shot.data["huge"] {
        assert_eq!(huge_bits.len(), 1000);
        assert!(huge_bits[0]);
        assert!(huge_bits[100]);
        assert!(huge_bits[500]);
        assert!(huge_bits[999]);
        assert_eq!(huge_bits.count_ones(), 4);
    }

    // Check test results
    if let Data::BitVec(test_bits) = &shot.data["test"] {
        assert_eq!(test_bits.len(), 10);

        assert!(test_bits[0], "test[0] should be 1");
        assert!(test_bits[1], "test[1] should be 1");
        assert!(test_bits[2], "test[2] should be 1");
        assert!(test_bits[3], "test[3] should be 1");

        assert!(!test_bits[4], "test[4] should be 0");
        assert!(!test_bits[5], "test[5] should be 0");

        assert!(test_bits[6], "test[6] should be 1");
        assert!(test_bits[7], "test[7] should be 1");
        assert!(test_bits[8], "test[8] should be 1");

        println!("Successfully tested edge cases with 1000-bit register!");
    }
}
