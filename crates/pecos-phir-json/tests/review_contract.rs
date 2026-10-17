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
fn result_rejects_non_string_arguments() {
    for value in [json!(7), json!(null), json!(["q", 0]), json!({"bad":true})] {
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
