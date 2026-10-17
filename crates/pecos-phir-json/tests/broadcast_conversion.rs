use pecos_engines::{ClassicalEngine, Engine, StateVecEngine};
use pecos_phir::{
    PhirEngine,
    ops::{Operation, QuantumOp},
};
use pecos_phir_json::{PhirJsonEngine, phir_json_to_module};
use serde_json::{Value, json};

fn program(ops: Vec<Value>) -> String {
    let mut all = vec![
        json!({"data":"qvar_define","data_type":"qubits","variable":"q","size":4}),
        json!({"data":"cvar_define","data_type":"u32","variable":"m","size":4}),
    ];
    all.extend(ops);
    // Reversed destinations make loss of result/target correspondence visible.
    all.push(json!({"qop":"Measure","args":[["q",0],["q",1],["q",2],["q",3]],"returns":[["m",3],["m",2],["m",1],["m",0]]}));
    all.push(json!({"cop":"Result","args":["m"],"returns":["out"]}));
    json!({"format":"PHIR/JSON","version":"0.1.0","ops":all}).to_string()
}
fn run(engine: &mut dyn ClassicalEngine) -> pecos_engines::Shot {
    let mut quantum = StateVecEngine::new(4);
    for _ in 0..20 {
        let commands = engine.generate_commands().unwrap();
        if commands.is_empty().unwrap() {
            return engine.get_results().unwrap();
        }
        let outcomes = quantum.process(commands).unwrap();
        engine.handle_measurements(outcomes).unwrap();
    }
    panic!("program did not terminate")
}
#[test]
fn broadcast_execution_matches_native() {
    let targets = json!([["q", 0], ["q", 2]]);
    let pairs = json!([[["q", 0], ["q", 1]], [["q", 2], ["q", 3]]]);
    let cases = [
        vec![json!({"qop":"X","args":targets})],
        vec![
            json!({"qop":"H","args":targets}),
            json!({"qop":"Z","args":[["q",0]]}),
            json!({"qop":"Z","args":[["q",2]]}),
            json!({"qop":"H","args":targets}),
        ],
        vec![
            json!({"qop":"H","args":[["q",0]]}),
            json!({"qop":"H","args":[["q",2]]}),
            json!({"qop":"RZ","args":targets,"angles":[[180.0],"deg"]}),
            json!({"qop":"H","args":[["q",0]]}),
            json!({"qop":"H","args":[["q",2]]}),
        ],
        vec![
            json!({"qop":"X","args":targets}),
            json!({"qop":"CX","args":pairs}),
        ],
        vec![
            json!({"qop":"H","args":targets}),
            json!({"qop":"SZZ","args":pairs}),
            json!({"qop":"SZZ","args":pairs}),
            json!({"qop":"H","args":targets}),
        ],
        vec![
            json!({"qop":"X","args":targets}),
            json!({"qop":"H","args":[["q",1],["q",3]]}),
            json!({"qop":"CZ","args":pairs}),
            json!({"qop":"H","args":[["q",1],["q",3]]}),
        ],
        vec![
            json!({"qop":"X","args":targets}),
            json!({"qop":"Init","args":targets}),
        ],
    ];
    for ops in cases {
        let source = program(ops);
        let mut native = PhirJsonEngine::from_json(&source).unwrap();
        let module = phir_json_to_module(&source).unwrap();
        for inst in &module.body.blocks[0].operations {
            if let Operation::Quantum(op) = &inst.operation {
                assert_eq!(Some(inst.operands.len()), op.operand_count());
            }
        }
        let mut converted = PhirEngine::new(module).unwrap();
        assert_eq!(
            run(&mut converted).data["out"].as_u32(),
            run(&mut native).data["out"].as_u32(),
            "{source}"
        );
    }
}
#[test]
fn cz_splits_pairs_in_order() {
    let source = program(vec![
        json!({"qop":"CZ","args":[[["q",2],["q",0]],[["q",3],["q",1]]]}),
    ]);
    let module = phir_json_to_module(&source).unwrap();
    let operands: Vec<_> = module.body.blocks[0]
        .operations
        .iter()
        .filter(|i| i.operation == Operation::Quantum(QuantumOp::CZ))
        .map(|i| i.operands.iter().map(|v| v.id).collect::<Vec<_>>())
        .collect();
    assert_eq!(operands, vec![vec![2, 0], vec![3, 1]]);
}

#[test]
fn every_converted_gate_has_canonical_broadcast_arity() {
    for (gate, arity, angles) in [
        ("H", 1, vec![]),
        ("X", 1, vec![]),
        ("Y", 1, vec![]),
        ("Z", 1, vec![]),
        ("S", 1, vec![]),
        ("Sdg", 1, vec![]),
        ("T", 1, vec![]),
        ("Tdg", 1, vec![]),
        ("SX", 1, vec![]),
        ("SXdg", 1, vec![]),
        ("Init", 1, vec![]),
        ("Prep", 1, vec![]),
        ("RZ", 1, vec![0.3]),
        ("R1XY", 1, vec![0.3, 0.1]),
        ("U", 1, vec![0.3, 0.1, 0.2]),
        ("CX", 2, vec![]),
        ("CZ", 2, vec![]),
        ("SXX", 2, vec![]),
        ("SXXdg", 2, vec![]),
        ("SYY", 2, vec![]),
        ("SYYdg", 2, vec![]),
        ("SZZ", 2, vec![]),
        ("SZZdg", 2, vec![]),
        ("RXYXY2Q", 2, vec![0.3, 0.1]),
    ] {
        let args = if arity == 1 {
            json!([["q", 2], ["q", 0]])
        } else {
            json!([[["q", 2], ["q", 0]], [["q", 3], ["q", 1]]])
        };
        let mut operation = json!({"qop":gate,"args":args});
        if !angles.is_empty() {
            operation["angles"] = json!([angles, "rad"]);
        }
        let module = phir_json_to_module(&program(vec![operation])).unwrap();
        let instructions: Vec<_> = module.body.blocks[0].operations.iter().filter(|inst| matches!(inst.operation, Operation::Quantum(ref op) if *op != QuantumOp::Measure)).collect();
        assert_eq!(instructions.len(), 2, "{gate}");
        assert!(
            instructions
                .iter()
                .all(|inst| inst.operands.len() == arity && inst.results.len() == arity),
            "{gate}"
        );
        assert_eq!(
            instructions[0].operation, instructions[1].operation,
            "angles must be identical for {gate}"
        );
    }
}

#[test]
fn cvar_export_and_result_have_different_native_selection_semantics() {
    // The specification makes cvar_export selective; the current native engine
    // ignores it. Replacing it with Result changes the observable result keys.
    let source = |export| {
        json!({"format":"PHIR/JSON","version":"0.1.0","ops":[
            {"data":"qvar_define","data_type":"qubits","variable":"q","size":4},
            {"data":"cvar_define","data_type":"u32","variable":"m","size":1},
            {"data":"cvar_define","data_type":"u32","variable":"scratch","size":1},
            {"qop":"X","args":[["q",0]]},
            {"qop":"Measure","args":[["q",0]],"returns":[["m",0]]},
            export
        ]})
        .to_string()
    };
    let with_export = source(json!({"data":"cvar_export","variables":["m"]}));
    let with_result = source(json!({"cop":"Result","args":["m"],"returns":["m"]}));
    let exported = run(&mut PhirJsonEngine::from_json(&with_export).unwrap());
    let mapped = run(&mut PhirJsonEngine::from_json(&with_result).unwrap());
    let converted = run(&mut PhirEngine::new(phir_json_to_module(&with_result).unwrap()).unwrap());
    assert_eq!(exported.data["m"].as_u32(), Some(1));
    assert!(exported.data.contains_key("scratch"));
    assert_eq!(
        mapped.data.keys().collect::<Vec<_>>(),
        vec!["_width_m", "m"]
    );
    assert_eq!(converted.data.keys().collect::<Vec<_>>(), vec!["m"]);
    assert_eq!(mapped.data["m"].as_u32(), converted.data["m"].as_u32());
    let error = phir_json_to_module(&with_export).unwrap_err().to_string();
    assert!(
        error.contains("data cvar_export") && error.contains("JSON-to-module"),
        "{error}"
    );
}

#[test]
fn result_pairs_preserve_values_names_and_order() {
    for result in [
        json!({"cop":"Result","args":["a"],"returns":["x"]}),
        json!({"cop":"Result","args":["a","b"],"returns":["x","y"]}),
        json!({"cop":"Result","args":["a","b"],"returns":["b","x"]}),
    ] {
        let source = json!({"format":"PHIR/JSON","version":"0.1.0","ops":[
            {"data":"qvar_define","data_type":"qubits","variable":"q","size":4},
            {"data":"cvar_define","data_type":"u32","variable":"a","size":2},
            {"data":"cvar_define","data_type":"u32","variable":"b","size":2},
            {"qop":"X","args":[["q",0]]},
            {"qop":"Measure","args":[["q",0],["q",1]],"returns":[["a",1],["b",0]]}, result
        ]})
        .to_string();
        let native = run(&mut PhirJsonEngine::from_json(&source).unwrap());
        let module = phir_json_to_module(&source).unwrap();
        let results: Vec<_> = module.body.blocks[0]
            .operations
            .iter()
            .filter(|i| i.operation == Operation::Classical(pecos_phir::ops::ClassicalOp::Result))
            .collect();
        assert_eq!(results.len(), result["args"].as_array().unwrap().len());
        assert!(results.iter().all(|i| i.operands.len() == 1));
        let converted = run(&mut PhirEngine::new(module).unwrap());
        for name in result["returns"].as_array().unwrap() {
            let name = name.as_str().unwrap();
            assert_eq!(
                converted.data[name].as_u32(),
                native.data[name].as_u32(),
                "{source}"
            );
        }
    }
}

#[test]
fn result_snapshots_do_not_move_past_later_measurements() {
    let source = json!({"format":"PHIR/JSON","version":"0.1.0","ops":[
        {"data":"qvar_define","data_type":"qubits","variable":"q","size":4},
        {"data":"cvar_define","data_type":"u32","variable":"a","size":2},
        {"qop":"X","args":[["q",0]]},
        {"qop":"Measure","args":[["q",0]],"returns":[["a",0]]},
        {"cop":"Result","args":["a"],"returns":["before"]},
        {"qop":"X","args":[["q",0]]},
        {"qop":"Measure","args":[["q",0]],"returns":[["a",0]]},
        {"cop":"Result","args":["a"],"returns":["after"]}
    ]})
    .to_string();
    let native = run(&mut PhirJsonEngine::from_json(&source).unwrap());
    let converted = run(&mut PhirEngine::new(phir_json_to_module(&source).unwrap()).unwrap());
    for (name, expected) in [("before", 1), ("after", 0)] {
        assert_eq!(native.data[name].as_u32(), Some(expected));
        assert_eq!(converted.data[name].as_u32(), Some(expected));
    }
}

#[test]
fn result_copies_preserve_widths_and_later_bit_writes() {
    for (dtype, bit, size) in [("u32", 1, 4), ("u64", 40, 64)] {
        let source = json!({"format":"PHIR/JSON","version":"0.1.0","ops":[
            {"data":"qvar_define","data_type":"qubits","variable":"q","size":4},
            {"data":"cvar_define","data_type":dtype,"variable":"a","size":size},
            {"data":"cvar_define","data_type":"u32","variable":"small","size":1},
            {"qop":"X","args":[["q",0]]},
            {"qop":"Measure","args":[["q",0]],"returns":[["a",bit]]},
            {"cop":"Result","args":["a","a"],"returns":["saved","small"]},
            {"qop":"Measure","args":[["q",0]],"returns":[["saved",0]]}
        ]})
        .to_string();
        let native = run(&mut PhirJsonEngine::from_json(&source).unwrap());
        let converted = run(&mut PhirEngine::new(phir_json_to_module(&source).unwrap()).unwrap());
        let expected = (1_u64 << bit) | 1;
        let actual = match converted.data["saved"] {
            pecos_engines::Data::U32(value) => u64::from(value),
            pecos_engines::Data::U64(value) => value,
            ref other => panic!("expected unsigned register, got {other:?}"),
        };
        assert_eq!(actual, expected);
        let native_bits = native.register_to_binary_string("saved").unwrap();
        assert_eq!(u64::from_str_radix(&native_bits, 2).unwrap(), expected);
        assert_eq!(converted.data["small"].as_u32(), Some(0));
    }
}
