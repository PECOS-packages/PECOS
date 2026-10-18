use pecos_engines::ClassicalEngine;
use pecos_phir_json::{PhirJsonEngine, phir_json_to_module};
use serde_json::{Value, json};

fn program(op: &Value) -> String {
    json!({"format":"PHIR/JSON","version":"0.1.0","ops":[
        {"data":"qvar_define","data_type":"qubits","variable":"q","size":4}, op
    ]})
    .to_string()
}

#[test]
fn pair_group_sizes_agree_between_native_and_converter() {
    for name in ["CX", "CZ", "SZZ", "RZZ"] {
        for count in [0, 1, 3, 4] {
            let group: Vec<_> = (0..count).map(|q| json!(["q", q])).collect();
            let mut op = json!({"qop":name,"args":[group]});
            if name == "RZZ" {
                op["angles"] = json!([[0.2], "rad"]);
            }
            let source = program(&op);
            let native_error = PhirJsonEngine::from_json(&source)
                .unwrap()
                .generate_commands()
                .err()
                .expect("invalid groups must fail")
                .to_string();
            let converter_error = phir_json_to_module(&source).unwrap_err().to_string();
            assert_eq!(native_error, converter_error);
            assert!(
                native_error.contains(name) && native_error.contains("argument group"),
                "{native_error}"
            );
            assert!(
                native_error.contains("supply groups of 2 qubits"),
                "{native_error}"
            );
        }
    }
}

#[test]
fn converter_rejects_blocks_instead_of_losing_their_gates() {
    for (name, block) in [
        (
            "qparallel",
            json!({"block":"qparallel","ops":[{"qop":"X","args":[["q",0]]}]}),
        ),
        (
            "if",
            json!({"block":"if","condition":{"cop":"==","args":[1,1]},"true_branch":[{"qop":"X","args":[["q",0]]}]}),
        ),
    ] {
        let source = program(&block);
        let commands = PhirJsonEngine::from_json(&source)
            .unwrap()
            .generate_commands()
            .unwrap()
            .quantum_ops()
            .unwrap();
        assert_eq!(commands, vec![pecos_core::Gate::x(&[0])]);
        let error = phir_json_to_module(&source).unwrap_err().to_string();
        assert!(
            error.contains(&format!("PHIR block {name}:"))
                && error.contains("JSON-to-module path does not support"),
            "{error}"
        );
        assert!(error.contains("native PHIR-JSON engine"), "{error}");
    }
}

#[test]
fn converter_rejects_other_unimplemented_operation_kinds() {
    for (name, op) in [
        (
            "PHIR mop Idle:",
            json!({"mop":"Idle","args":[["q",0]],"duration":[1,"ns"]}),
        ),
        (
            "PHIR meta barrier:",
            json!({"meta":"barrier","args":[["q",0]]}),
        ),
        (
            "cvar_export",
            json!({"data":"cvar_export","variables":["m"]}),
        ),
        ("cop =", json!({"cop":"=","args":[1],"returns":["m"]})),
        ("unexpected", json!({"unexpected":true})),
        ("42", json!(42)),
    ] {
        let error = phir_json_to_module(&program(&op)).unwrap_err().to_string();
        assert!(error.contains(name), "{error}");
        if name == "cop =" {
            assert!(error.contains("JSON-to-module"), "{error}");
        }
        assert!(
            error.contains("use ") || error.contains("supply "),
            "{error}"
        );
    }
}

fn native_error(source: &str) -> String {
    let error = match PhirJsonEngine::from_json(source) {
        Ok(mut engine) => engine
            .generate_commands()
            .err()
            .expect("invalid qop must fail"),
        Err(error) => error,
    };
    assert!(
        matches!(error, pecos_core::errors::PecosError::Input(_)),
        "bad quantum arguments must be input errors: {error}"
    );
    error.to_string()
}

fn interpreter_error(source: &str) -> String {
    let mut interpreter =
        pecos_phir_json::v0_1::classical_interpreter::PhirClassicalInterpreter::new();
    match interpreter.init(source, None) {
        Ok(_) => interpreter.execute_program().unwrap_err().to_string(),
        Err(error) => error.to_string(),
    }
}

#[test]
fn all_quantum_paths_reject_duplicate_qubits_identically() {
    for (name, args) in [
        ("H", json!([["q", 0], ["q", 0]])),
        ("Measure", json!([["q", 1], ["q", 1]])),
        ("CX", json!([[["q", 0], ["q", 1]], [["q", 1], ["q", 2]]])),
    ] {
        let source = program(&json!({"qop":name,"args":args}));
        let error = native_error(&source);
        assert_eq!(error, interpreter_error(&source));
        assert_eq!(error, phir_json_to_module(&source).unwrap_err().to_string());
        assert!(error.contains("repeats qubit IDs"), "{error}");
    }
}

#[test]
fn all_quantum_paths_reject_malformed_groups_identically() {
    for name in ["CX", "CZ", "SZZ", "RZZ", "RXYXY2Q"] {
        for count in [0, 1, 3, 4] {
            let group: Vec<_> = (0..count).map(|q| json!(["q", q])).collect();
            let mut op = json!({"qop":name,"args":[group]});
            if name == "RZZ" {
                op["angles"] = json!([[0.2], "rad"]);
            }
            if name == "RXYXY2Q" {
                op["angles"] = json!([[0.2, 0.3], "rad"]);
            }
            let source = program(&op);
            let error = native_error(&source);
            assert_eq!(error, phir_json_to_module(&source).unwrap_err().to_string());
            assert_eq!(error, interpreter_error(&source));
            assert!(
                error.contains("argument group") && error.contains("supply groups of 2 qubits"),
                "{error}"
            );
        }
    }
}

#[test]
fn rxyxy2q_requires_pairs_on_every_path() {
    for args in [
        json!([["q", 0], ["q", 1]]),
        json!([["q", 0], [["q", 1], ["q", 2]]]),
    ] {
        let source = program(&json!({"qop":"RXYXY2Q","args":args,"angles":[[0.2,0.3],"rad"]}));
        let error = native_error(&source);
        assert_eq!(error, phir_json_to_module(&source).unwrap_err().to_string());
        assert_eq!(error, interpreter_error(&source));
        assert!(error.contains("supply groups of 2 qubits"), "{error}");
    }
}

#[test]
fn result_rejects_invalid_references() {
    for value in [
        json!(7),
        json!(null),
        json!(["q", "bad"]),
        json!({"bad":true}),
    ] {
        let source = program(&json!({"cop":"Result","args":[value],"returns":["out"]}));
        let error = phir_json_to_module(&source).unwrap_err().to_string();
        assert!(
            error.contains("Result")
                && error.contains(&value.to_string())
                && error.contains("supply"),
            "{error}"
        );
    }
}

#[test]
fn result_rejects_non_string_destinations_and_non_array_fields() {
    for (field, value) in [
        ("returns", json!([7])),
        ("returns", json!("out")),
        ("args", json!("q")),
        ("args", json!(null)),
    ] {
        let mut op = json!({"cop":"Result","args":["q"],"returns":["out"]});
        op[field] = value.clone();
        let error = phir_json_to_module(&program(&op)).unwrap_err().to_string();
        assert!(
            error.contains("Result") && error.contains(field) && error.contains("supply"),
            "{error}"
        );
    }
}

#[test]
fn all_quantum_paths_reject_invalid_signatures_identically() {
    for op in [
        json!({"qop":"UnknownGate","args":[["q",0]]}),
        json!({"qop":"H","args":[["q",0]],"angles":[[0.1],"rad"]}),
        json!({"qop":"RXYXY2Q","args":[[["q",0],["q",1]]],"angles":[[0.1],"rad"]}),
        json!({"qop":"H","args":[]}),
        json!({"qop":"RXYXY2Q","args":[],"angles":[[0.1,0.2],"rad"]}),
    ] {
        let source = program(&op);
        let error = native_error(&source);
        assert_eq!(error, phir_json_to_module(&source).unwrap_err().to_string());
        assert_eq!(error, interpreter_error(&source));
        assert!(
            error.contains("supply") || error.contains("use "),
            "{error}"
        );
    }
}

#[test]
fn resolution_errors_precede_signature_errors_on_all_paths() {
    let source = program(&json!({"qop":"H","args":[["q",9]],"angles":[[0.2],"rad"]}));
    let error = native_error(&source);
    assert_eq!(error, interpreter_error(&source));
    assert_eq!(error, phir_json_to_module(&source).unwrap_err().to_string());
    assert!(error.contains('9') && error.contains("size 4"), "{error}");
}

#[test]
fn cphase_is_not_a_phir_json_gate() {
    let source =
        program(&json!({"qop":"CPhase","args":[[["q",0],["q",1]]],"angles":[[0.2],"rad"]}));
    let error = native_error(&source);
    assert_eq!(error, interpreter_error(&source));
    assert_eq!(error, phir_json_to_module(&source).unwrap_err().to_string());
    assert!(error.contains("CPhase") && error.contains("Unsupported quantum gate"));
}

#[test]
fn result_counts_are_validated_before_conversion_or_execution() {
    for (args, returns) in [
        (json!(["a"]), json!([])),
        (json!([]), json!([])),
        (json!(["a", "a"]), json!(["x"])),
        (json!(["a"]), json!(["x", "y"])),
    ] {
        let source = json!({"format":"PHIR/JSON","version":"0.1.0","ops":[
            {"data":"cvar_define","data_type":"u32","variable":"a","size":2},
            {"cop":"Result","args":args,"returns":returns}
        ]})
        .to_string();
        let error = phir_json_to_module(&source).unwrap_err().to_string();
        assert_eq!(error, native_error(&source));
        assert_eq!(error, interpreter_error(&source));
        assert!(
            error.contains("Result") && error.contains("supply") && error.contains("counts"),
            "{error}"
        );
    }
}

#[test]
fn measurement_destinations_are_declared_and_within_declared_size() {
    use pecos_phir_json::v0_1::classical_interpreter::{MeasKey, PhirClassicalInterpreter};
    for (name, dtype, bit, message) in [
        (
            "s",
            "i32",
            3,
            "Measurement write s[3] is out of range for a register of declared size 3",
        ),
        (
            "s",
            "u32",
            3,
            "Measurement write s[3] is out of range for a register of declared size 3",
        ),
        (
            "missing",
            "u32",
            0,
            "unknown classical variable \"missing\"",
        ),
    ] {
        let source = json!({"format":"PHIR/JSON","version":"0.1.0","ops":[
            {"data":"qvar_define","variable":"q","size":1},
            {"data":"cvar_define","data_type":dtype,"variable":"s","size":3},
            {"qop":"Measure","args":[["q",0]],"returns":[[name,bit]]}
        ]})
        .to_string();
        let mut native = PhirJsonEngine::from_json(&source).unwrap();
        let native_error = match native.generate_commands() {
            Err(error) => error,
            Ok(commands) => {
                use pecos_engines::Engine;
                let outcomes = pecos_engines::StateVecEngine::new(1)
                    .process(commands)
                    .unwrap();
                native.handle_measurements(outcomes).unwrap_err()
            }
        }
        .to_string();
        let mut interpreter = PhirClassicalInterpreter::new();
        interpreter.init(&source, None).unwrap();
        let error = interpreter
            .receive_results(&[[(MeasKey::Bit(name.into(), bit), 1)].into()])
            .unwrap_err()
            .to_string();
        assert!(error.contains(message), "{error}");
        assert_eq!(error, native_error);
        assert_eq!(error, phir_json_to_module(&source).unwrap_err().to_string());
    }
}

#[test]
fn result_quantum_destinations_have_the_same_named_error() {
    let source = json!({"format":"PHIR/JSON","version":"0.1.0","ops":[
        {"data":"qvar_define","variable":"q","size":1},
        {"data":"cvar_define","data_type":"u32","variable":"a","size":3},
        {"cop":"Result","args":["a"],"returns":["q"]}
    ]})
    .to_string();
    let error = native_error(&source);
    assert!(
        error.contains("Result: destination \"q\" is quantum"),
        "{error}"
    );
    assert_eq!(error, interpreter_error(&source));
    assert_eq!(error, phir_json_to_module(&source).unwrap_err().to_string());
}

#[test]
fn result_reference_errors_agree_across_rust_paths() {
    for (args, returns, message) in [
        (
            json!(["unknown"]),
            json!(["out"]),
            "unknown classical variable \"unknown\"",
        ),
        (json!([["a", 3]]), json!(["out"]), "bit index 3"),
        (json!(["a"]), json!([["a", 3]]), "bit index 3"),
        (json!(["a"]), json!([["out", 64]]), "bit index 64"),
        (
            json!(["a"]),
            json!([["q", 0]]),
            "destination \"q\" is quantum",
        ),
    ] {
        let source = json!({"format":"PHIR/JSON","version":"0.1.0","ops":[
            {"data":"qvar_define","variable":"q","size":1},
            {"data":"cvar_define","data_type":"i32","variable":"a","size":3},
            {"cop":"Result","args":args,"returns":returns}
        ]})
        .to_string();
        let error = native_error(&source);
        assert!(error.contains(message), "{error}");
        assert_eq!(error, interpreter_error(&source));
        assert_eq!(error, phir_json_to_module(&source).unwrap_err().to_string());
    }
}

#[test]
fn bool_size_is_rejected_at_declaration() {
    let source =
        program(&json!({"data":"cvar_define","data_type":"bool","variable":"flag","size":4}));
    let error = native_error(&source);
    assert!(
        error.contains("bool") && error.contains("size 4") && error.contains("size 1"),
        "{error}"
    );
    assert_eq!(error, interpreter_error(&source));
    assert_eq!(error, phir_json_to_module(&source).unwrap_err().to_string());
}

#[test]
fn malformed_result_entries_have_identical_json_errors() {
    for entry in [
        json!(7),
        json!(null),
        json!(true),
        json!({"bad":true}),
        json!(["a", -1]),
        json!(["a", true]),
        json!(["a", 1.5]),
        json!(["a"]),
    ] {
        for field in ["args", "returns"] {
            let mut op = json!({"cop":"Result","args":["a"],"returns":["out"]});
            op[field] = json!([entry]);
            let source = json!({"format":"PHIR/JSON","version":"0.1.0","ops":[
                {"data":"cvar_define","data_type":"u32","variable":"a","size":3}, op
            ]})
            .to_string();
            let error = match PhirJsonEngine::from_json(&source) {
                Err(error) => error.to_string(),
                Ok(mut engine) => engine
                    .generate_commands()
                    .err()
                    .expect("invalid Result must fail")
                    .to_string(),
            };
            assert!(
                error.contains(&format!("Result: invalid {field} entry {entry}")),
                "{error}"
            );
            assert_eq!(error, interpreter_error(&source));
            assert_eq!(error, phir_json_to_module(&source).unwrap_err().to_string());
        }
    }
}

#[test]
fn assignment_and_ffcall_validate_all_destinations_before_execution() {
    use pecos_phir_json::v0_1::{
        ast::ArgItem, classical_interpreter::PhirClassicalInterpreter,
        operations::OperationProcessor,
    };
    for cop in ["=", "ffcall"] {
        for target in [
            json!(["a", 5]),
            json!(["a", 3]),
            json!(["missing", 0]),
            json!("missing"),
            json!(["非ASCII", 0]),
        ] {
            let returns: Vec<ArgItem> = serde_json::from_value(json!([target])).unwrap();
            let source = json!({"format":"PHIR/JSON","version":"0.1.0","ops":[
                {"data":"cvar_define","data_type":"u32","variable":"a","size":3}
            ]})
            .to_string();
            let mut native = OperationProcessor::new();
            native.add_classical_variable("a", "u32", 3).unwrap();
            let native_error = native
                .handle_classical_op(cop, &[ArgItem::Integer(1)], &returns, &[], 0)
                .unwrap_err()
                .to_string();
            let mut interpreter = PhirClassicalInterpreter::new();
            interpreter.init(&source, None).unwrap();
            let error = interpreter
                .handle_cop(cop, &[ArgItem::Integer(1)], &returns, &Some("f".into()))
                .unwrap_err()
                .to_string();
            assert_eq!(error, native_error);
            assert!(
                error.contains("declare it") || error.contains("use an index below 3"),
                "{error}"
            );
        }
    }
}

#[test]
fn result_and_assignment_sign_extend_identically() {
    use pecos_phir_json::v0_1::{
        ast::ArgItem,
        classical_interpreter::{PhirClassicalInterpreter, ResultValue},
        operations::OperationProcessor,
    };
    for (dtype, max_size) in [("i8", 7), ("i16", 15), ("i32", 31), ("i64", 63)] {
        for size in [3, max_size] {
            for destination in ["i64", "u64"] {
                let destination_size = if destination == "i64" { 63 } else { 64 };
                let source = json!({"format":"PHIR/JSON","version":"0.1.0","ops":[
                    {"data":"cvar_define","data_type":dtype,"variable":"s","size":size},
                    {"data":"cvar_define","data_type":destination,"variable":"assigned","size":destination_size},
                    {"data":"cvar_define","data_type":destination,"variable":"copied","size":destination_size}
                ]}).to_string();
                let mut native = OperationProcessor::new();
                native.add_classical_variable("s", dtype, size).unwrap();
                for name in ["assigned", "copied"] {
                    native
                        .add_classical_variable(name, destination, destination_size)
                        .unwrap();
                }
                let mut interpreter = PhirClassicalInterpreter::new();
                interpreter.init(&source, None).unwrap();
                for (cop, args, returns) in [
                    (
                        "=",
                        vec![ArgItem::Integer(-8)],
                        vec![ArgItem::Simple("s".into())],
                    ),
                    (
                        "=",
                        vec![ArgItem::Simple("s".into())],
                        vec![ArgItem::Simple("assigned".into())],
                    ),
                    (
                        "Result",
                        vec![ArgItem::Simple("s".into())],
                        vec![ArgItem::Simple("copied".into())],
                    ),
                ] {
                    native
                        .handle_classical_op(cop, &args, &returns, &[], 0)
                        .unwrap();
                    interpreter.handle_cop(cop, &args, &returns, &None).unwrap();
                }
                for name in ["assigned", "copied"] {
                    assert_eq!(
                        native.environment.get(name).unwrap().as_u64(),
                        u64::MAX - 7,
                        "{dtype} {size} {destination}"
                    );
                    match &interpreter.results(true)[name] {
                        ResultValue::Int(value, _) => assert_eq!(*value, -8),
                        ResultValue::UInt(value, _) => assert_eq!(*value, u64::MAX - 7),
                        other @ ResultValue::BitString(_) => panic!("unexpected {other:?}"),
                    }
                }
            }
        }
    }
}

#[test]
fn malformed_result_references_are_input_errors() {
    use pecos_core::errors::PecosError;
    for entry in [json!(7), json!(["a", -1]), json!(["a", true])] {
        let source = program(&json!({"cop":"Result","args":[entry],"returns":["out"]}));
        let error = PhirJsonEngine::from_json(&source).unwrap_err();
        assert!(matches!(error, PecosError::Input(_)), "{error:?}");
        assert!(
            error
                .to_string()
                .starts_with("Input error: Result: invalid args entry")
        );
        assert!(matches!(
            phir_json_to_module(&source),
            Err(PecosError::Input(_))
        ));
    }
}

#[test]
fn assignment_unknown_target_names_the_operation() {
    use pecos_phir_json::v0_1::{ast::ArgItem, operations::OperationProcessor};
    let mut processor = OperationProcessor::new();
    let error = processor
        .handle_classical_op(
            "=",
            &[ArgItem::Integer(1)],
            &[ArgItem::Simple("非ASCII".into())],
            &[],
            0,
        )
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "Input error: Assignment: unknown classical variable \"非ASCII\"; declare it with cvar_define before use"
    );
}

#[test]
fn converter_rejects_present_noninteger_register_sizes() {
    let mut failures = Vec::new();
    for data in ["cvar_define", "qvar_define"] {
        for size in [json!(-1), json!("8"), json!(1.5), json!(true)] {
            let source = json!({"format":"PHIR/JSON","version":"0.1.0","ops":[{
                "data": data, "data_type": if data == "cvar_define" { "u32" } else { "qubits" },
                "variable": "bad_size", "size": size
            }]})
            .to_string();
            let expected = format!(
                "Register 'bad_size' has invalid size {size}; use a non-negative integer size"
            );
            match phir_json_to_module(&source) {
                Err(pecos_core::errors::PecosError::Input(message)) if message == expected => {}
                result => failures.push(format!("{data} size={size}: {result:?}")),
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn declaration_size_invalid_values_agree() {
    use pecos_phir_json::v0_1::{
        ast::Operation, classical_interpreter::PhirClassicalInterpreter,
        classical_validation::validate_classical_inputs,
    };
    let mut failures = Vec::new();
    for (data, dtype) in [
        ("cvar_define", "u32"),
        ("cvar_define", "bool"),
        ("qvar_define", "qubits"),
    ] {
        for size in [json!(-1), json!("8"), json!(1.5), json!(true)] {
            let declaration = json!({"data":data,"data_type":dtype,"variable":"a","size":size});
            let source = json!({"format":"PHIR/JSON","version":"0.1.0","ops":[declaration]});
            let encoded = source.to_string();
            let expected = format!(
                "Input error: Register 'a' has invalid size {size}; use a non-negative integer size"
            );
            let mut interpreter = PhirClassicalInterpreter::new();
            for (path, result) in [
                ("native", PhirJsonEngine::from_json(&encoded).map(|_| ())),
                ("interpreter", interpreter.init(&encoded, None).map(|_| ())),
                ("converter", phir_json_to_module(&encoded).map(|_| ())),
                ("validation", validate_classical_inputs(&source)),
            ] {
                println!("{dtype} size={size} {path}: {result:?}");
                if !matches!(&result, Err(error) if error.to_string() == expected) {
                    failures.push(format!("{dtype} size={size} {path}: {result:?}"));
                }
            }
            // Exercise D::Error::custom independently of shared program validation.
            let ast_error = serde_json::from_value::<Operation>(declaration)
                .err()
                .map(|error| error.to_string());
            if ast_error.as_deref() != Some(&expected) {
                failures.push(format!("{dtype} size={size} AST: {ast_error:?}"));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn null_declaration_sizes_match_omission() {
    use pecos_phir_json::v0_1::{
        classical_interpreter::PhirClassicalInterpreter,
        classical_validation::validate_classical_inputs,
    };
    let mut failures = Vec::new();
    for (data, dtype) in [
        ("cvar_define", "u32"),
        ("cvar_define", "bool"),
        ("qvar_define", "qubits"),
    ] {
        let omitted = json!({"format":"PHIR/JSON","version":"0.1.0","ops":[
            {"data":data,"data_type":dtype,"variable":"a"}
        ]});
        let mut explicit_null = omitted.clone();
        explicit_null["ops"][0]["size"] = Value::Null;
        for source in [&omitted, &explicit_null] {
            if let Err(error) = validate_classical_inputs(source) {
                failures.push(format!(
                    "{dtype} validation rejected omission/null: {error}"
                ));
            }
        }
        let run = |source: &Value| {
            let mut interpreter = PhirClassicalInterpreter::new();
            let encoded = source.to_string();
            [
                ("native", PhirJsonEngine::from_json(&encoded).map(|_| ())),
                ("interpreter", interpreter.init(&encoded, None).map(|_| ())),
                ("converter", phir_json_to_module(&encoded).map(|_| ())),
            ]
            .map(|(path, result)| (path, result.map_err(|error| error.to_string())))
        };
        let before = run(&omitted);
        let after = run(&explicit_null);
        if before != after {
            failures.push(format!("{dtype} omitted={before:?}, null={after:?}"));
        }
        for (path, result) in after {
            if data == "qvar_define" {
                // Null is absent, so the existing required-size rule still applies.
                if !matches!(&result, Err(error) if error.contains("Quantum register 'a' requires a size"))
                {
                    failures.push(format!("{path}: {result:?}"));
                }
            } else if let Err(error) = result {
                failures.push(format!("{dtype} {path}: {error}"));
            }
        }
        if data == "cvar_define" {
            let a = phir_json_to_module(&omitted.to_string());
            let b = phir_json_to_module(&explicit_null.to_string());
            if let (Ok(a), Ok(b)) = (a, b) {
                assert_eq!(a, b, "null and omitted sizes must infer identical modules");
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
