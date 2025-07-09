#[cfg(feature = "wasm")]
mod wasm_tests {
    use pecos_qasm::simulation::qasm_sim;
    use std::io::Write;
    use std::path::PathBuf;

    #[test]
    fn test_wasm_addition() {
        let qasm = r"
            OPENQASM 2.0;
            creg a[10];
            creg b[10];
            creg c[10];
            a = 1;
            b = 2;
            c = add(a, b);
        ";

        let wat_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("wat")
            .join("add.wat");

        let results = qasm_sim(qasm)
            .wasm(wat_path.to_string_lossy())
            .run(100)
            .expect("Simulation should succeed");

        // Check that all shots have the expected values
        for shot in &results.shots {
            let a_val = shot.data.get("a").expect("Register 'a' should exist");
            let b_val = shot.data.get("b").expect("Register 'b' should exist");
            let c_val = shot.data.get("c").expect("Register 'c' should exist");

            // Convert to integers for comparison
            let a_int = match a_val {
                pecos_engines::shot_results::Data::U64(v) => *v,
                pecos_engines::shot_results::Data::BigInt(v) => {
                    v.to_string().parse::<u64>().unwrap()
                }
                pecos_engines::shot_results::Data::BitVec(bv) => {
                    // Convert BitVec to u64
                    let mut value = 0u64;
                    for (i, bit) in bv.iter().enumerate() {
                        if i >= 64 {
                            break;
                        }
                        if *bit {
                            value |= 1u64 << i;
                        }
                    }
                    value
                }
                _ => panic!("Expected U64, BigInt, or BitVec for register a, got: {a_val:?}"),
            };
            let b_int = match b_val {
                pecos_engines::shot_results::Data::U64(v) => *v,
                pecos_engines::shot_results::Data::BigInt(v) => {
                    v.to_string().parse::<u64>().unwrap()
                }
                pecos_engines::shot_results::Data::BitVec(bv) => {
                    // Convert BitVec to u64
                    let mut value = 0u64;
                    for (i, bit) in bv.iter().enumerate() {
                        if i >= 64 {
                            break;
                        }
                        if *bit {
                            value |= 1u64 << i;
                        }
                    }
                    value
                }
                _ => panic!("Expected U64, BigInt, or BitVec for register b"),
            };
            let c_int = match c_val {
                pecos_engines::shot_results::Data::U64(v) => *v,
                pecos_engines::shot_results::Data::BigInt(v) => {
                    v.to_string().parse::<u64>().unwrap()
                }
                pecos_engines::shot_results::Data::BitVec(bv) => {
                    // Convert BitVec to u64
                    let mut value = 0u64;
                    for (i, bit) in bv.iter().enumerate() {
                        if i >= 64 {
                            break;
                        }
                        if *bit {
                            value |= 1u64 << i;
                        }
                    }
                    value
                }
                _ => panic!("Expected U64, BigInt, or BitVec for register c"),
            };

            assert_eq!(a_int, 1, "Register a should be 1");
            assert_eq!(b_int, 2, "Register b should be 2");
            assert_eq!(c_int, 3, "Register c should be 3 (1 + 2)");
        }
    }

    #[test]
    fn test_wasm_cannot_override_builtin() {
        // Test that built-in functions cannot be overridden by WASM functions
        // This test uses sin() in a gate parameter where it's valid
        let qasm = r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            rz(sin(1)) q[0];  // This should use the built-in sin, not WASM
        "#;

        let wat_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("wat")
            .join("add.wat");

        // Even with WASM loaded, built-in functions should not be overridden
        let result = qasm_sim(qasm).wasm(wat_path.to_string_lossy()).run(1);

        // This should succeed as it uses the built-in sin function
        assert!(
            result.is_ok(),
            "Built-in sin function should work with WASM loaded"
        );
    }

    #[test]
    fn test_wasm_validation_missing_function() {
        let qasm = r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            creg c[1];
            creg result[8];
            result = multiply(5, 6);
        "#;

        let wat_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("wat")
            .join("missing_func.wat");

        let result = qasm_sim(qasm).wasm(wat_path.to_string_lossy()).build();

        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(
            err.to_string()
                .contains("Function 'multiply' is called in QASM but not exported")
        );
        assert!(err.to_string().contains("Available functions: [\"init\"]"));
    }

    #[test]
    fn test_wasm_validation_missing_init() {
        // Create a WAT file without init function
        let wat_content = r#"
            (module
              (func $add (export "add") (param i32 i32) (result i32)
                local.get 0
                local.get 1
                i32.add
              )
            )
        "#;

        // Write to a temporary file
        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        temp_file.write_all(wat_content.as_bytes()).unwrap();
        temp_file.flush().unwrap();

        let qasm = r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            creg c[1];
        "#;

        let result = qasm_sim(qasm)
            .wasm(temp_file.path().to_string_lossy())
            .build();

        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(
            err.to_string()
                .contains("WebAssembly module must export an 'init' function")
        );
    }

    #[test]
    fn test_wasm_with_quantum_operations() {
        let qasm = r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            creg sum[10];

            h q[0];
            cx q[0], q[1];
            measure q -> c;

            sum = add(c[0], c[1]);
        "#;

        let wat_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("wat")
            .join("add.wat");

        let results = qasm_sim(qasm)
            .wasm(wat_path.to_string_lossy())
            .run(1000)
            .expect("Simulation should succeed");

        // Verify quantum entanglement and WASM addition
        for shot in &results.shots {
            let c_val = shot.data.get("c").expect("Register 'c' should exist");
            let sum_val = shot.data.get("sum").expect("Register 'sum' should exist");

            // Convert to integers
            let c_int = match c_val {
                pecos_engines::shot_results::Data::U64(v) => *v,
                pecos_engines::shot_results::Data::BigInt(v) => {
                    v.to_string().parse::<u64>().unwrap()
                }
                pecos_engines::shot_results::Data::BitVec(bv) => {
                    // Convert BitVec to u64
                    let mut value = 0u64;
                    for (i, bit) in bv.iter().enumerate() {
                        if i >= 64 {
                            break;
                        }
                        if *bit {
                            value |= 1u64 << i;
                        }
                    }
                    value
                }
                _ => panic!("Expected U64, BigInt, or BitVec for register c"),
            };
            let sum_int = match sum_val {
                pecos_engines::shot_results::Data::U64(v) => *v,
                pecos_engines::shot_results::Data::BigInt(v) => {
                    v.to_string().parse::<u64>().unwrap()
                }
                pecos_engines::shot_results::Data::BitVec(bv) => {
                    // Convert BitVec to u64
                    let mut value = 0u64;
                    for (i, bit) in bv.iter().enumerate() {
                        if i >= 64 {
                            break;
                        }
                        if *bit {
                            value |= 1u64 << i;
                        }
                    }
                    value
                }
                _ => panic!("Expected U64, BigInt, or BitVec for register sum"),
            };

            // Due to entanglement, c should be either 0 (00) or 3 (11)
            assert!(
                c_int == 0 || c_int == 3,
                "c should be 0 or 3 due to entanglement"
            );

            // sum should be 0 (0+0) or 2 (1+1)
            if c_int == 0 {
                assert_eq!(sum_int, 0, "sum should be 0 when c is 0");
            } else {
                assert_eq!(sum_int, 2, "sum should be 2 when c is 3");
            }
        }
    }

    #[test]
    fn test_wasm_void_function() {
        // Test that void functions work (functions with no return value)
        let qasm = r"
            OPENQASM 2.0;
            creg a[10];
            creg b[10];
            a = 5;
            b = 10;
            void_func(a, b);  // Now we support standalone void function calls!
        ";

        let wat_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("wat")
            .join("multiple_funcs.wat");

        let result = qasm_sim(qasm).wasm(wat_path.to_string_lossy()).run(1);

        assert!(result.is_ok(), "Void functions should work");
    }

    #[test]
    fn test_wasm_multiple_functions() {
        // Test using multiple WASM functions in one program
        let qasm = r"
            OPENQASM 2.0;
            creg a[10];
            creg b[10];
            creg c[10];
            creg d[10];
            a = 5;
            b = 3;
            c = add(a, b);        // c = 8
            d = multiply(c, 2);   // d = 16
            a = negate(b);        // a = -3 (but stored as two's complement)
        ";

        let wat_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("wat")
            .join("multiple_funcs.wat");

        let results = qasm_sim(qasm)
            .wasm(wat_path.to_string_lossy())
            .run(10)
            .expect("Simulation should succeed");

        for shot in &results.shots {
            let c_val = shot.data.get("c").expect("Register 'c' should exist");
            let d_val = shot.data.get("d").expect("Register 'd' should exist");

            let c_int = extract_u64(c_val);
            let d_int = extract_u64(d_val);

            assert_eq!(c_int, 8, "c should be 8 (5+3)");
            assert_eq!(d_int, 16, "d should be 16 (8*2)");
        }
    }

    #[test]
    fn test_wasm_sequential_function_calls() {
        // Test sequential function calls (instead of nested)
        let qasm = r"
            OPENQASM 2.0;
            creg a[10];
            creg temp1[10];
            creg temp2[10];
            creg result[10];
            a = 3;
            temp1 = multiply(a, 2);     // 3 * 2 = 6
            temp2 = add(4, 1);          // 4 + 1 = 5
            result = add(temp1, temp2); // 6 + 5 = 11
        ";

        let wat_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("wat")
            .join("multiple_funcs.wat");

        let results = qasm_sim(qasm)
            .wasm(wat_path.to_string_lossy())
            .run(10)
            .expect("Simulation should succeed");

        for shot in &results.shots {
            let result_val = shot
                .data
                .get("result")
                .expect("Register 'result' should exist");
            let result_int = extract_u64(result_val);
            assert_eq!(result_int, 11, "result should be 11");
        }
    }

    #[test]
    fn test_wasm_state_reset_between_shots() {
        // Test that init() is called between shots, resetting state
        let qasm = r"
            OPENQASM 2.0;
            creg counter1[10];
            creg counter2[10];
            counter1 = increment();  // Should always be 1
            counter2 = increment();  // Should always be 2
        ";

        let wat_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("wat")
            .join("stateful.wat");

        let results = qasm_sim(qasm)
            .wasm(wat_path.to_string_lossy())
            .run(10)
            .expect("Simulation should succeed");

        // Each shot should have the same values because init() resets state
        for shot in &results.shots {
            let c1 = extract_u64(shot.data.get("counter1").unwrap());
            let c2 = extract_u64(shot.data.get("counter2").unwrap());
            assert_eq!(c1, 1, "counter1 should always be 1 (init resets state)");
            assert_eq!(c2, 2, "counter2 should always be 2 (init resets state)");
        }
    }

    #[test]
    fn test_wasm_large_values() {
        // Test handling of large values and i64
        let qasm = r"
            OPENQASM 2.0;
            creg a[64];
            creg b[64];
            creg c[64];
            a = 1000000000;
            b = 2000000000;
            c = add64(a, b);  // 3 billion - needs i64
        ";

        let wat_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("wat")
            .join("multiple_funcs.wat");

        let results = qasm_sim(qasm)
            .wasm(wat_path.to_string_lossy())
            .run(1)
            .expect("Simulation should succeed");

        let shot = &results.shots[0];
        let c_val = shot.data.get("c").expect("Register 'c' should exist");
        let c_int = extract_u64(c_val);
        assert_eq!(c_int, 3_000_000_000, "c should be 3 billion");
    }

    #[test]
    fn test_wasm_conditional_execution() {
        // Test WASM functions in conditional statements
        let qasm = r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            creg c[1];
            creg result[10];

            h q[0];
            measure q[0] -> c[0];

            if(c==1) result = add(10, 20);
            if(c==0) result = multiply(5, 6);
        "#;

        let wat_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("wat")
            .join("multiple_funcs.wat");

        let results = qasm_sim(qasm)
            .wasm(wat_path.to_string_lossy())
            .run(1000)
            .expect("Simulation should succeed");

        let mut saw_add = false;
        let mut saw_multiply = false;

        for shot in &results.shots {
            let c_val = extract_u64(shot.data.get("c").unwrap());
            let result_val = extract_u64(shot.data.get("result").unwrap());

            if c_val == 1 {
                assert_eq!(result_val, 30, "When c=1, result should be 30");
                saw_add = true;
            } else {
                assert_eq!(result_val, 30, "When c=0, result should be 30");
                saw_multiply = true;
            }
        }

        assert!(saw_add && saw_multiply, "Should see both branches execute");
    }

    #[test]
    fn test_wasm_function_with_computed_args() {
        // Test WASM functions with pre-computed arguments
        let qasm = r"
            OPENQASM 2.0;
            creg a[10];
            creg b[10];
            creg c[10];
            creg temp1[10];
            creg temp2[10];
            creg result[10];
            a = 5;
            b = 3;
            c = 2;
            temp1 = a + b;           // 5 + 3 = 8
            temp2 = multiply(c, 4);  // 2 * 4 = 8
            result = add(temp1, temp2);  // 8 + 8 = 16
        ";

        let wat_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("wat")
            .join("multiple_funcs.wat");

        let results = qasm_sim(qasm)
            .wasm(wat_path.to_string_lossy())
            .run(1)
            .expect("Simulation should succeed");

        let shot = &results.shots[0];
        let result_val = extract_u64(shot.data.get("result").unwrap());
        assert_eq!(result_val, 16, "result should be 16");
    }

    // Helper function to extract u64 from Data enum
    fn extract_u64(data: &pecos_engines::shot_results::Data) -> u64 {
        use pecos_engines::shot_results::Data;
        match data {
            Data::U64(v) => *v,
            Data::BigInt(v) => v.to_string().parse::<u64>().unwrap(),
            Data::BitVec(bv) => {
                // Use bitvec's built-in conversion
                let bytes = bv.as_raw_slice();
                if bytes.is_empty() {
                    0
                } else {
                    // Take first 8 bytes (64 bits) and convert to u64
                    let mut result = 0u64;
                    for (i, &byte) in bytes.iter().take(8).enumerate() {
                        result |= u64::from(byte) << (i * 8);
                    }
                    result
                }
            }
            _ => panic!("Expected U64, BigInt, or BitVec"),
        }
    }
}
