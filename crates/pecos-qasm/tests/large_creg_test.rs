use pecos_engines::{Data, sim_builder};
use pecos_programs::Qasm;
use pecos_qasm::qasm_engine;

#[test]
fn test_large_classical_register() {
    // Test with a large classical register but small quantum register
    // This demonstrates that BitVec can handle registers larger than 64 bits
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[4];
        creg c[128];  // 128-bit classical register

        // Set some bits by measuring qubits
        x q[0];
        measure q[0] -> c[0];

        x q[1];
        measure q[1] -> c[63];

        x q[2];
        measure q[2] -> c[64];  // Beyond 64-bit boundary

        x q[3];
        measure q[3] -> c[127];
    "#;

    let shot_vec = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .run(1)
        .unwrap();
    let shot = &shot_vec.shots[0];

    // Check that we have the correct register
    assert!(shot.data.contains_key("c"));

    if let Data::BitVec(bitvec) = &shot.data["c"] {
        // Verify the register has 128 bits
        assert_eq!(bitvec.len(), 128);

        // Check the bits we set
        assert!(bitvec[0]); // bit 0 should be 1
        assert!(bitvec[63]); // bit 63 should be 1
        assert!(bitvec[64]); // bit 64 should be 1 (beyond 64-bit boundary)
        assert!(bitvec[127]); // bit 127 should be 1

        // Check some unset bits
        assert!(!bitvec[1]);
        assert!(!bitvec[62]);
        assert!(!bitvec[65]);
        assert!(!bitvec[126]);

        // Count total set bits
        let ones_count = bitvec.count_ones();
        assert_eq!(ones_count, 4);

        println!("Successfully handled 128-bit classical register!");
    } else {
        panic!("Expected BitVec data type");
    }
}

#[test]
fn test_very_large_classical_register() {
    // Test with a 256-bit classical register
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[5];
        creg c[256];  // 256-bit classical register

        // Set bits at various positions beyond 64-bit boundary
        x q[0];
        measure q[0] -> c[0];

        x q[1];
        measure q[1] -> c[100];

        x q[2];
        measure q[2] -> c[200];

        x q[3];
        measure q[3] -> c[255];
    "#;

    let shot_vec = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .run(1)
        .unwrap();
    let shot = &shot_vec.shots[0];

    if let Data::BitVec(bitvec) = &shot.data["c"] {
        assert_eq!(bitvec.len(), 256);
        assert!(bitvec[0]);
        assert!(bitvec[100]);
        assert!(bitvec[200]);
        assert!(bitvec[255]);

        // Count total set bits
        let ones_count = bitvec.count_ones();
        assert_eq!(ones_count, 4);

        println!("Successfully handled 256-bit classical register!");
    }
}

#[test]
fn test_classical_assignment_beyond_64_bits() {
    // Test classical assignment with bit indices beyond 64-bit boundary
    let qasm = r"
        OPENQASM 2.0;

        creg c[80];

        // Set individual bits beyond 64-bit boundary
        c[0] = 1;
        c[63] = 1;
        c[64] = 1;
        c[70] = 1;
        c[79] = 1;
    ";

    let shot_vec = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .run(1)
        .unwrap();
    let shot = &shot_vec.shots[0];

    if let Data::BitVec(bitvec) = &shot.data["c"] {
        assert_eq!(bitvec.len(), 80);
        assert!(bitvec[0]);
        assert!(bitvec[63]);
        assert!(bitvec[64]);
        assert!(bitvec[70]);
        assert!(bitvec[79]);

        // All other bits should be 0
        for i in 1..63 {
            assert!(!bitvec[i]);
        }
        for i in 65..70 {
            assert!(!bitvec[i]);
        }
        for i in 71..79 {
            assert!(!bitvec[i]);
        }

        println!("Successfully handled bit assignments beyond 64-bit boundary!");
    }
}

#[test]
fn test_large_register_arithmetic() {
    // Test that arithmetic operations work correctly with large registers
    let qasm = r"
        OPENQASM 2.0;

        creg c[72];  // 72-bit register

        // Set bits to create a pattern
        c[0] = 1;
        c[1] = 1;
        c[8] = 1;
        c[16] = 1;
        c[32] = 1;
        c[64] = 1;  // Beyond 64-bit boundary
        c[71] = 1;
    ";

    let shot_vec = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .run(1)
        .unwrap();
    let shot = &shot_vec.shots[0];

    if let Data::BitVec(bitvec) = &shot.data["c"] {
        assert_eq!(bitvec.len(), 72);

        // Check the pattern
        assert!(bitvec[0]);
        assert!(bitvec[1]);
        assert!(bitvec[8]);
        assert!(bitvec[16]);
        assert!(bitvec[32]);
        assert!(bitvec[64]);
        assert!(bitvec[71]);

        let ones_count = bitvec.count_ones();
        assert_eq!(ones_count, 7);

        println!("Successfully handled large register arithmetic!");
    }
}

#[test]
fn test_register_value_assignment_limitation() {
    // Test that direct value assignment is limited to 64 bits
    // This is a current limitation in the expression evaluator
    let qasm = r"
        OPENQASM 2.0;

        creg c[80];

        // This will only set the lower 63 bits due to i64 limitation
        c = 9223372036854775807;  // 2^63 - 1 (max i64 value)
    ";

    let shot_vec = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .run(1)
        .unwrap();
    let shot = &shot_vec.shots[0];

    if let Data::BitVec(bitvec) = &shot.data["c"] {
        assert_eq!(bitvec.len(), 80);

        // First 63 bits should be 1
        for i in 0..63 {
            assert!(bitvec[i], "Bit {i} should be 1");
        }

        // Bit 63 and beyond should be 0 (current limitation with signed i64)
        for i in 63..80 {
            assert!(!bitvec[i], "Bit {i} should be 0");
        }

        println!("Demonstrated current 64-bit limitation in value assignment");
    }
}
