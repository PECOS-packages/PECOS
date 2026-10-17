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
            error.contains(name) && error.contains("JSON-to-module path does not support"),
            "{error}"
        );
        assert!(error.contains("native PHIR-JSON engine"), "{error}");
    }
}

#[test]
fn converter_rejects_other_unimplemented_operation_kinds() {
    for (name, op) in [
        (
            "Idle",
            json!({"mop":"Idle","args":[["q",0]],"duration":[1,"ns"]}),
        ),
        ("barrier", json!({"meta":"barrier","args":[["q",0]]})),
        (
            "cvar_export",
            json!({"data":"cvar_export","variables":["m"]}),
        ),
        ("=", json!({"cop":"=","args":[1],"returns":["m"]})),
        ("unexpected", json!({"unexpected":true})),
        ("42", json!(42)),
    ] {
        let error = phir_json_to_module(&program(&op)).unwrap_err().to_string();
        assert!(error.contains(name), "{error}");
        assert!(
            error.contains("use ") || error.contains("supply "),
            "{error}"
        );
    }
}
