use pecos_engines::ClassicalControlEngineBuilder;
use pecos_programs::Qasm;
use pecos_qasm::qasm_engine;

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

    let results = qasm_engine()
        .program(Qasm::from_string(qasm))
        .to_sim()
        .seed(42)
        .workers(1)
        .run(10)
        .unwrap();

    assert_eq!(results.len(), 10);
    assert!(results.shots[0].data.contains_key("c"));

    // Check that all results are either 0 or 3 for Bell state
    // (either 00 or 11 in binary, which is 0 or 3 in decimal)
    let mut has_zero = false;
    let mut has_three = false;

    for shot in &results.shots {
        let value = shot
            .data
            .get("c")
            .and_then(pecos_engines::prelude::Data::as_u32)
            .expect("c register should be convertible to u32");
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

    let results = qasm_engine()
        .program(Qasm::from_string(qasm))
        .to_sim()
        .seed(42)
        .workers(1)
        .run(10)
        .unwrap();

    assert!(
        results.shots[0].data.contains_key("d"),
        "Results should contain 'd' register"
    );
    assert_eq!(results.len(), 10, "Expected 10 measurement results");

    // Check all shots have d register set to 1
    for shot in &results.shots {
        let value = shot
            .data
            .get("d")
            .and_then(pecos_engines::prelude::Data::as_u32)
            .expect("d register should be convertible to u32");
        assert_eq!(value, 1, "Expected measurement result to be 1");
    }
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

    let results = qasm_engine()
        .program(Qasm::from_string(qasm))
        .to_sim()
        .seed(42)
        .workers(1)
        .run(10)
        .unwrap();

    println!("Arbitrary register test results: {results:?}");

    // Assert that arbitrary register name exists in results
    assert!(
        results.shots[0].data.contains_key("result"),
        "Results should contain 'result' register"
    );

    // Assert that we have exactly 10 shots
    assert_eq!(results.len(), 10, "Expected 10 measurement results");

    // Check that all results are either 0 or 3 for Bell state
    // (either 00 or 11 in binary, which is 0 or 3 in decimal)
    for shot in &results.shots {
        let value = shot
            .data
            .get("result")
            .and_then(pecos_engines::prelude::Data::as_u32)
            .expect("result register should be convertible to u32");
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

    let results = qasm_engine()
        .program(Qasm::from_string(qasm))
        .to_sim()
        .seed(42)
        .workers(1)
        .run(10)
        .unwrap();

    assert!(
        results.shots[0].data.contains_key("c"),
        "Results should contain 'c' register"
    );
    assert!(
        results.shots[0].data.contains_key("d"),
        "Results should contain 'd' register"
    );

    assert_eq!(results.len(), 10, "Expected 10 measurement results");

    // Check all shots have expected values
    for shot in &results.shots {
        let c_value = shot
            .data
            .get("c")
            .and_then(pecos_engines::prelude::Data::as_u32)
            .expect("c register should be convertible to u32");
        let d_value = shot
            .data
            .get("d")
            .and_then(pecos_engines::prelude::Data::as_u32)
            .expect("d register should be convertible to u32");
        assert_eq!(c_value, 3, "Expected c measurement result to be 3");
        assert_eq!(d_value, 4, "Expected d measurement result to be 4");
    }
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

    let results = qasm_engine()
        .program(Qasm::from_string(qasm))
        .to_sim()
        .seed(42)
        .workers(1)
        .run(10)
        .unwrap();

    println!("Arithmetic test results: {results:?}");

    assert!(
        results.shots[0].data.contains_key("a"),
        "Results should contain 'a' register"
    );
    assert!(
        results.shots[0].data.contains_key("b"),
        "Results should contain 'b' register"
    );

    assert_eq!(results.len(), 10, "Expected 10 shots");

    // Test that arithmetic worked correctly - all 'a' values should be 3 (1+2)
    // 'b' values should be 0
    for shot in &results.shots {
        let a_value = shot
            .data
            .get("a")
            .and_then(pecos_engines::prelude::Data::as_u32)
            .expect("a register should be convertible to u32");
        let b_value = shot
            .data
            .get("b")
            .and_then(pecos_engines::prelude::Data::as_u32)
            .expect("b register should be convertible to u32");
        assert_eq!(a_value, 3, "Expected all 'a' results to be 3 (1+2)");
        assert_eq!(b_value, 0, "Expected all 'b' results to be 0");
    }
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

    let results = qasm_engine()
        .program(Qasm::from_string(qasm))
        .to_sim()
        .seed(42)
        .workers(1)
        .run(5)
        .unwrap();

    println!("Default test results: {results:?}");

    assert!(
        results.shots[0].data.contains_key("a"),
        "Results should contain 'a' register"
    );
    assert!(
        results.shots[0].data.contains_key("b"),
        "Results should contain 'b' register"
    );
    assert!(
        results.shots[0].data.contains_key("m"),
        "Results should contain 'm' register"
    );

    assert_eq!(results.len(), 5);

    // Check all shots have expected default values (0)
    for shot in &results.shots {
        let a_value = shot
            .data
            .get("a")
            .and_then(pecos_engines::prelude::Data::as_u32)
            .expect("a register should be convertible to u32");
        let b_value = shot
            .data
            .get("b")
            .and_then(pecos_engines::prelude::Data::as_u32)
            .expect("b register should be convertible to u32");
        let m_value = shot
            .data
            .get("m")
            .and_then(pecos_engines::prelude::Data::as_u32)
            .expect("m register should be convertible to u32");
        assert_eq!(a_value, 0);
        assert_eq!(b_value, 0);
        assert_eq!(m_value, 0);
    }
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

    let results = qasm_engine()
        .program(Qasm::from_string(qasm))
        .to_sim()
        .seed(42)
        .workers(1)
        .run(10)
        .unwrap();

    println!("If creg test results: {results:?}");

    assert!(
        results.shots[0].data.contains_key("a"),
        "Results should contain 'a' register"
    );
    assert!(
        results.shots[0].data.contains_key("b"),
        "Results should contain 'b' register"
    );

    assert_eq!(results.len(), 10, "Expected 10 shots");

    // Test that arithmetic worked correctly - all 'a' values should be 3 (1+2)
    // 'b' values should be 0
    for shot in &results.shots {
        let a_value = shot
            .data
            .get("a")
            .and_then(pecos_engines::prelude::Data::as_u32)
            .expect("a register should be convertible to u32");
        let b_value = shot
            .data
            .get("b")
            .and_then(pecos_engines::prelude::Data::as_u32)
            .expect("b register should be convertible to u32");
        assert_eq!(a_value, 3, "Expected all 'a' results to be 3 (1+2)");
        assert_eq!(b_value, 0, "Expected all 'b' results to be 0");
    }
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

    let results = qasm_engine()
        .program(Qasm::from_string(qasm))
        .to_sim()
        .seed(42)
        .workers(1)
        .run(10)
        .unwrap();

    println!("If creg test results: {results:?}");

    assert!(
        results.shots[0].data.contains_key("a"),
        "Results should contain 'a' register"
    );
    assert!(
        results.shots[0].data.contains_key("b"),
        "Results should contain 'b' register"
    );

    assert_eq!(results.len(), 10, "Expected 10 shots");

    // Check all shots have expected values
    for shot in &results.shots {
        let a_value = shot
            .data
            .get("a")
            .and_then(pecos_engines::prelude::Data::as_u32)
            .expect("a register should be convertible to u32");
        let b_value = shot
            .data
            .get("b")
            .and_then(pecos_engines::prelude::Data::as_u32)
            .expect("b register should be convertible to u32");
        assert_eq!(a_value, 2); // Value 2 = binary 10 (bit 1 = 1, bit 0 = 0)
        assert_eq!(b_value, 0);
    }
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

    let results = qasm_engine()
        .program(Qasm::from_string(qasm))
        .to_sim()
        .seed(42)
        .workers(1)
        .run(10)
        .unwrap();

    println!("Conditional test results: {results:?}");

    assert!(results.shots[0].data.contains_key("one_0"));

    // Check all shots have one_0 register set to 2
    for shot in &results.shots {
        let value = shot
            .data
            .get("one_0")
            .and_then(pecos_engines::prelude::Data::as_u32)
            .expect("one_0 register should be convertible to u32");
        assert_eq!(value, 2);
    }
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

    let results = qasm_engine()
        .program(Qasm::from_string(qasm))
        .to_sim()
        .seed(42)
        .workers(1)
        .run(10)
        .unwrap();

    println!("Conditional test results: {results:?}");

    assert!(results.shots[0].data.contains_key("c"));

    // Check all shots have c register set to 2
    for shot in &results.shots {
        let value = shot
            .data
            .get("c")
            .and_then(pecos_engines::prelude::Data::as_u32)
            .expect("c register should be convertible to u32");
        assert_eq!(value, 2);
    }
}
