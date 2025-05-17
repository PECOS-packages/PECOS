#[path = "../helper.rs"]
mod helper;

use helper::run_qasm_sim;

#[test]
fn test_bell_qasm() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];

        // Bell state
        H q[0];
        CX q[0],q[1];
        measure q[0] -> c[0];
        measure q[1] -> c[1];
    "#;

    let results = run_qasm_sim(qasm, 10, Some(42)).unwrap();

    assert!(results.contains_key("c"));
    assert_eq!(results["c"].len(), 10);

    // Check that all results are either 0 or 3 for Bell state
    // (either 00 or 11 in binary, which is 0 or 3 in decimal)
    let mut has_zero = false;
    let mut has_three = false;

    for &value in &results["c"] {
        println!("Checking value: {value}");
        assert!(
            value == 0 || value == 3,
            "Expected value to be 0 or 3, but got {value}"
        );

        // Track if we've seen both expected values
        if value == 0 {
            has_zero = true;
        }
        if value == 3 {
            has_three = true;
        }
    }

    // Assert that we observed both possible outcomes at least once
    assert!(has_zero, "Expected at least one '0' outcome but found none");
    assert!(
        has_three,
        "Expected at least one '3' outcome but found none"
    );
}

#[test]
fn test_x_qasm() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg w[1];
        creg d[1];

        X w[0];
        measure w[0] -> d[0];
    "#;

    let results = run_qasm_sim(qasm, 10, Some(42)).unwrap();

    assert!(
        results.contains_key("d"),
        "Results should contain 'd' register"
    );
    assert_eq!(results["d"].len(), 10, "Expected 10 measurement results");

    let expected = vec![1u32; 10];
    assert_eq!(
        results["d"], expected,
        "Expected all measurement results to be 1"
    );
}

#[test]
fn test_arbitrary_register_names() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        // Arbitrary register names
        qreg alice[1];
        qreg bob[1];
        creg result[2];

        // Bell state with arbitrary register names
        H alice[0];
        CX alice[0],bob[0];
        measure alice[0] -> result[0];
        measure bob[0] -> result[1];
    "#;

    let results = run_qasm_sim(qasm, 10, Some(42)).unwrap();

    println!("Arbitrary register test results: {results:?}");

    // Assert that arbitrary register name exists in results
    assert!(
        results.contains_key("result"),
        "Results should contain 'result' register"
    );

    // Assert that "result" has exactly 10 elements
    assert_eq!(
        results["result"].len(),
        10,
        "Expected 10 measurement results"
    );

    // Check that all results are either 0 or 3 for Bell state
    // (either 00 or 11 in binary, which is 0 or 3 in decimal)
    for &value in &results["result"] {
        assert!(
            value == 0 || value == 3,
            "Expected value to be 0 or 3, but got {value}"
        );
    }
}

#[test]
fn test_flips_multi_reg_qasm() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg a[3];
        qreg b[3];

        creg c[3];
        creg d[3];

        X a[0];
        X a[1];

        X b[2];

        measure a -> c;
        measure b -> d;
    "#;

    let results = run_qasm_sim(qasm, 10, Some(42)).unwrap();

    assert!(
        results.contains_key("c"),
        "Results should contain 'c' register"
    );
    assert!(
        results.contains_key("d"),
        "Results should contain 'd' register"
    );

    assert_eq!(results["c"].len(), 10, "Expected 10 measurement results");
    assert_eq!(results["d"].len(), 10, "Expected 10 measurement results");

    let expected = vec![3; 10];
    assert_eq!(
        results["c"], expected,
        "Expected all measurement results to be 3"
    );

    let expected = vec![4; 10];
    assert_eq!(
        results["d"], expected,
        "Expected all measurement results to be 4"
    );
}

#[test]
fn test_basic_arthmetic_qasm() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[1];

        creg a[3];
        creg b[3];

        // Now we can use arithmetic operations directly
        a = 1 + 2;
        b = 0;
    "#;

    let results = run_qasm_sim(qasm, 10, Some(42)).unwrap();

    println!("Arithmetic test results: {results:?}");

    assert!(
        results.contains_key("a"),
        "Results should contain 'a' register"
    );
    assert!(
        results.contains_key("b"),
        "Results should contain 'b' register"
    );

    assert_eq!(
        results["a"].len(),
        10,
        "Expected 10 measurement results for 'a'"
    );
    assert_eq!(
        results["b"].len(),
        10,
        "Expected 10 measurement results for 'b'"
    );

    // Test that arithmetic worked correctly - all 'a' values should be 3 (1+2)
    let expected_a = vec![3u32; 10]; // Vector of 10 elements, all set to 3u32
    assert_eq!(
        results["a"], expected_a,
        "Expected all 'a' results to be 3 (1+2)"
    );

    // 'b' values should be 1 in bit 0 (from the x gate and measurement)
    let expected_b = vec![0u32; 10]; // Vector of 10 elements, all set to 1u32
    assert_eq!(
        results["b"], expected_b,
        "Expected all 'b' results to be 1 at bit 0"
    );
}

#[test]
fn test_defaults_qasm() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[1];

        creg a[3];
        creg b[3];
        creg m[1];

        measure q -> m;
    "#;

    let results = run_qasm_sim(qasm, 5, Some(42)).unwrap();

    println!("Default test results: {results:?}");

    assert!(
        results.contains_key("a"),
        "Results should contain 'a' register"
    );
    assert!(
        results.contains_key("b"),
        "Results should contain 'b' register"
    );
    assert!(
        results.contains_key("m"),
        "Results should contain 'm' register"
    );

    assert_eq!(results["a"].len(), 5);
    assert_eq!(results["b"].len(), 5);
    assert_eq!(results["m"].len(), 5);

    let expected = vec![0; 5];
    assert_eq!(results["a"], expected);
    assert_eq!(results["b"], expected);
    assert_eq!(results["m"], expected);
}

#[test]
fn test_basic_if_creg_statements_qasm() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[1];

        creg a[3];
        creg b[3];

        if(b==0) a = 1 + 2;
    "#;

    let results = run_qasm_sim(qasm, 10, Some(42)).unwrap();

    println!("If creg test results: {results:?}");

    assert!(
        results.contains_key("a"),
        "Results should contain 'a' register"
    );
    assert!(
        results.contains_key("b"),
        "Results should contain 'b' register"
    );

    assert_eq!(
        results["a"].len(),
        10,
        "Expected 10 measurement results for 'a'"
    );
    assert_eq!(
        results["b"].len(),
        10,
        "Expected 10 measurement results for 'b'"
    );

    // Test that arithmetic worked correctly - all 'a' values should be 3 (1+2)
    let expected_a = vec![3u32; 10]; // Vector of 10 elements, all set to 3u32
    assert_eq!(
        results["a"], expected_a,
        "Expected all 'a' results to be 3 (1+2)"
    );

    // 'b' values should be 1 in bit 0 (from the x gate and measurement)
    let expected_b = vec![0u32; 10]; // Vector of 10 elements, all set to 1u32
    assert_eq!(
        results["b"], expected_b,
        "Expected all 'b' results to be 1 at bit 0"
    );
}

#[test]
fn test_basic_if_qreg_statements_qasm() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[1];

        creg a[2];
        creg b[3];

        if(b==0) X q[0];

        // Let's measure both qubits so we can verify the conditional operation
        measure q[0] -> a[1];
    "#;

    let results = run_qasm_sim(qasm, 10, Some(42)).unwrap();

    println!("If creg test results: {results:?}");

    assert!(
        results.contains_key("a"),
        "Results should contain 'a' register"
    );
    assert!(
        results.contains_key("b"),
        "Results should contain 'b' register"
    );

    assert_eq!(
        results["a"].len(),
        10,
        "Expected 10 measurement results for 'a'"
    );
    assert_eq!(
        results["b"].len(),
        10,
        "Expected 10 measurement results for 'b'"
    );

    let expected_a = vec![2u32; 10]; // Value 2 = binary 10 (bit 1 = 1, bit 0 = 0)
    assert_eq!(results["a"], expected_a);

    let expected_b = vec![0u32; 10];
    assert_eq!(results["b"], expected_b);
}

#[test]
fn test_cond_bell() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg one_0[2];

        // Bell state
        H q[0];
        CX q[0],q[1];
        measure q[0] -> one_0[0]; // collapses to 00 or 11

        // use the measurement of the other qubit to flip deterministically to |1>
        if(one_0[0]==0) X q[1];

        // one_0[1] should always be 1
        measure q[1] -> one_0[1];
        one_0[0] = 0; // reset first bit to 0
        // c should be "10" == 2
    "#;

    let results = run_qasm_sim(qasm, 10, Some(42)).unwrap();

    println!("Conditional test results: {results:?}");

    assert!(results.contains_key("one_0"));
    let expected_b = vec![2u32; 10];
    assert_eq!(results["one_0"], expected_b);
}

#[test]
fn test_classical_statement() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg m[32];
        creg a[32];
        creg b[32];
        creg c[32];

        b = 2;

        X q[0];
        measure q[0] -> m[0];
        // m = 1;

        a = 2;

        // bit-wise XOR
        c = b ^ m;
        // "10" ^ "01" = "11" = 3

        // bit-wise OR
        c = c | 1;
        // "11" | "01" = "11" = 3
        c = c & a;
        // "11" & "10" = "10" = 2

    "#;

    let results = run_qasm_sim(qasm, 10, Some(42)).unwrap();

    println!("Conditional test results: {results:?}");

    assert!(results.contains_key("c"));
    let expected = vec![2u32; 10];
    assert_eq!(results["c"], expected);
}
