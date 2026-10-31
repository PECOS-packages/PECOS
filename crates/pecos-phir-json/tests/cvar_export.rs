//! End-to-end export selection, independently normalized across public result formats.
use pecos_engines::{ClassicalEngine, Data, Engine, Shot, StateVecEngine};
use pecos_phir::PhirEngine;
use pecos_phir_json::v0_1::{
    ast::PHIRProgram,
    block_executor::BlockExecutor,
    classical_interpreter::{MeasKey, PhirClassicalInterpreter, ResultValue, YieldedOp},
};
use pecos_phir_json::{PhirJsonEngine, phir_json_to_module};
use serde_json::{Value, json};
use std::collections::BTreeMap;

fn program(ops: Vec<Value>) -> String {
    let ops = Value::Array(ops);
    json!({"format":"PHIR/JSON","version":"0.1.0","ops":ops}).to_string()
}
fn declarations() -> Vec<Value> {
    vec![
        json!({"data":"qvar_define","variable":"q","size":1}),
        json!({"data":"cvar_define","variable":"m","data_type":"u32","size":2}),
        json!({"data":"cvar_define","variable":"a","data_type":"u32","size":2}),
        json!({"data":"cvar_define","variable":"scratch","data_type":"u32","size":2}),
    ]
}
fn export(variables: &[&str]) -> Value {
    json!({"data":"cvar_export","variables":variables})
}
fn renamed(source: &str, output: &str) -> Value {
    json!({"data":"cvar_export","variables":[source],"to":[output]})
}
fn run(engine: &mut dyn ClassicalEngine) -> Shot {
    let mut quantum = StateVecEngine::new(1);
    for _ in 0..20 {
        let commands = engine.generate_commands().unwrap();
        if commands.is_empty().unwrap() {
            return engine.get_results().unwrap();
        }
        engine
            .handle_measurements(quantum.process(commands).unwrap())
            .unwrap();
    }
    panic!("program did not terminate")
}
fn integer(data: &Data) -> u64 {
    match data {
        Data::U32(value) => u64::from(*value),
        Data::U64(value) => *value,
        Data::BitVec(bits) => bits
            .iter()
            .by_vals()
            .enumerate()
            .fold(0, |value, (bit, set)| value | (u64::from(set) << bit)),
        other => panic!("unexpected integer transport: {other:?}"),
    }
}
fn normalize(shot: &Shot, widths: &BTreeMap<String, usize>) -> BTreeMap<String, u64> {
    shot.data
        .iter()
        .filter(|(name, _)| !name.starts_with("_width_"))
        .map(|(name, value)| {
            let width = widths[name];
            let mask = u64::MAX >> (64 - width);
            (name.clone(), integer(value) & mask)
        })
        .collect()
}
fn normalize_interpreter(
    result: &BTreeMap<String, ResultValue>,
    widths: &BTreeMap<String, usize>,
) -> BTreeMap<String, u64> {
    result
        .iter()
        .map(|(name, value)| {
            let raw = match value {
                ResultValue::UInt(value, _) => *value,
                ResultValue::Int(value, _) => u64::from_ne_bytes(value.to_ne_bytes()),
                ResultValue::BitString(bits) => u64::from_str_radix(bits, 2).unwrap(),
            };
            (name.clone(), raw & (u64::MAX >> (64 - widths[name])))
        })
        .collect()
}
fn expected(entries: &[(&str, u64, usize)]) -> (BTreeMap<String, u64>, BTreeMap<String, usize>) {
    (
        entries.iter().map(|(n, v, _)| ((*n).into(), *v)).collect(),
        entries.iter().map(|(n, _, w)| ((*n).into(), *w)).collect(),
    )
}
fn assert_native(shot: &Shot, entries: &[(&str, u64, usize)]) {
    let (values, widths) = expected(entries);
    assert_eq!(normalize(shot, &widths), values);
    assert_eq!(shot.data.len(), entries.len() * 2);
    for (name, _, width) in entries {
        assert_eq!(
            integer(&shot.data[&format!("_width_{name}")]),
            *width as u64
        );
        assert!(matches!(&shot.data[*name], Data::BitVec(bits) if bits.len() == *width));
    }
}
fn execute_interpreter(interpreter: &mut PhirClassicalInterpreter) {
    // Fixtures use X then measurements of q[0]; inject that deterministic outcome
    // at each yield boundary, before the next classical operation executes.
    let ops = interpreter.program_ops().to_vec();
    for op in &ops {
        let batches = {
            let mut iter = interpreter.execute_ops(std::slice::from_ref(op));
            let mut batches = Vec::new();
            while let Some(batch) = iter.next_batch().unwrap() {
                batches.push(batch);
            }
            batches
        };
        for batch in batches {
            for op in batch {
                if let YieldedOp::QOp(op) = op
                    && let Some(returns) = op.returns
                {
                    interpreter
                        .receive_results(&[returns
                            .into_iter()
                            .map(|(n, i)| (MeasKey::Bit(n, i), 1))
                            .collect()])
                        .unwrap();
                }
            }
        }
    }
}
fn assert_interpreter(interpreter: &PhirClassicalInterpreter, entries: &[(&str, u64, usize)]) {
    let (values, widths) = expected(entries);
    for ints in [true, false] {
        let result = interpreter.results(ints).unwrap();
        assert_eq!(normalize_interpreter(&result, &widths), values);
        for (name, value) in &result {
            match value {
                ResultValue::BitString(bits) => {
                    assert!(!ints);
                    assert_eq!(bits.len(), widths[name]);
                }
                ResultValue::Int(_, _) | ResultValue::UInt(_, _) => assert!(ints),
            }
        }
    }
}
fn check_measurements(exports: Vec<Value>, entries: &[(&str, u64, usize)]) {
    let mut ops = declarations();
    // The directive precedes the writes, ruling out source-position snapshots.
    ops.extend(exports);
    ops.push(json!({"qop":"X","args":[["q",0]]}));
    for (name, bit) in [("m", 0), ("a", 1), ("scratch", 0)] {
        ops.push(json!({"qop":"Measure","args":[["q",0]],"returns":[[name,bit]]}));
    }
    let source = program(ops);
    let mut native = PhirJsonEngine::from_json(&source).unwrap();
    let mut converted = PhirEngine::new(phir_json_to_module(&source).unwrap()).unwrap();
    let mut interpreter = PhirClassicalInterpreter::new();
    interpreter.init(&source, None).unwrap();
    for _ in 0..2 {
        assert_native(&run(&mut native), entries);
        let shot = run(&mut converted);
        let (values, widths) = expected(entries);
        assert_eq!(normalize(&shot, &widths), values);
        assert_eq!(
            shot.data.len(),
            entries.len(),
            "module result has no width metadata"
        );
        execute_interpreter(&mut interpreter);
        assert_interpreter(&interpreter, entries);
        ClassicalEngine::reset(&mut native).unwrap();
        ClassicalEngine::reset(&mut converted).unwrap();
        interpreter.shot_reinit();
    }
}
#[test]
fn no_export_compatibility() {
    check_measurements(vec![], &[("a", 2, 2), ("m", 1, 2), ("scratch", 1, 2)]);
}
#[test]
fn subset() {
    check_measurements(vec![export(&["m", "a"])], &[("a", 2, 2), ("m", 1, 2)]);
}
#[test]
fn rename() {
    check_measurements(vec![renamed("m", "out")], &[("out", 1, 2)]);
}
#[test]
fn separate_exports() {
    check_measurements(
        vec![export(&["m"]), export(&["a"])],
        &[("a", 2, 2), ("m", 1, 2)],
    );
}
#[test]
fn export_and_result() {
    // Result precedes measurement: it must retain the copied zero value.
    check_measurements(
        vec![
            export(&["m"]),
            json!({"cop":"Result","args":["a"],"returns":["copy"]}),
        ],
        &[("copy", 0, 2), ("m", 1, 2)],
    );
}
#[test]
fn empty_export() {
    check_measurements(vec![export(&[])], &[]);
}
#[test]
fn empty_export_and_result() {
    check_measurements(
        vec![
            export(&[]),
            json!({"cop":"Result","args":["a"],"returns":["copy"]}),
        ],
        &[("copy", 0, 2)],
    );
}
#[test]
fn multiple_outputs() {
    check_measurements(
        vec![renamed("m", "out"), renamed("m", "other")],
        &[("other", 1, 2), ("out", 1, 2)],
    );
}
#[test]
fn duplicate_pair() {
    check_measurements(vec![export(&["m"]), export(&["m"])], &[("m", 1, 2)]);
}

fn check_classical(ops: Vec<Value>, entries: &[(&str, u64, usize)]) {
    let source = program(ops);
    let mut native = PhirJsonEngine::from_json(&source).unwrap();
    assert_native(&run(&mut native), entries);
    let mut interpreter = PhirClassicalInterpreter::new();
    interpreter.init(&source, None).unwrap();
    interpreter.execute_program().unwrap();
    assert_interpreter(&interpreter, entries);
    let ast = PHIRProgram::from_json(&source).unwrap();
    let output = BlockExecutor::new().execute_program(&ast.ops).unwrap();
    let (_, widths) = expected(entries);
    let normalized: BTreeMap<_, _> = output
        .into_iter()
        .map(|(n, v)| {
            let mask = u64::MAX >> (64 - widths[&n]);
            (n, v & mask)
        })
        .collect();
    assert_eq!(normalized, expected(entries).0);
}
#[test]
fn unexecuted_nested_export_and_result() {
    let mut ops = declarations();
    ops.push(
        json!({"block":"if","condition":{"cop":"==","args":[0,1]},"true_branch":[
        {"block":"sequence","ops":[export(&["m"])]},
        {"cop":"Result","args":["a"],"returns":["copy"]}
    ],"false_branch":[renamed("a","other")]}),
    );
    check_classical(ops, &[("m", 0, 2), ("other", 0, 2), ("copy", 0, 2)]);
}
#[test]
fn final_value_and_measurement_prefix() {
    for output in ["measurement_user", "out"] {
        let mut ops = declarations();
        ops.push(
            json!({"data":"cvar_define","variable":"measurement_user","data_type":"u32","size":2}),
        );
        ops.push(renamed("measurement_user", output));
        ops.push(json!({"cop":"=","args":[3],"returns":["measurement_user"]}));
        check_classical(ops, &[(output, 3, 2)]);
    }
}
#[test]
fn declaration_in_unexecuted_branch() {
    check_classical(
        vec![
            json!({"block":"if","condition":{"cop":"==","args":[0,1]},"true_branch":[
                {"data":"cvar_define","variable":"inner","data_type":"u32","size":4},
                renamed("inner","out")
            ]}),
        ],
        &[("out", 0, 4)],
    );
}
#[test]
fn signed_and_wide_sources() {
    check_classical(
        vec![
            json!({"data":"cvar_define","variable":"signed","data_type":"i32","size":4}),
            json!({"data":"cvar_define","variable":"wide","data_type":"u64","size":64}),
            json!({"data":"cvar_define","variable":"above31","data_type":"u32","size":32}),
            renamed("signed", "negative"),
            renamed("wide", "out"),
            export(&["above31"]),
            json!({"cop":"=","args":[-3],"returns":["signed"]}),
            json!({"cop":"=","args":[9_223_372_036_854_775_809_u64],"returns":["wide"]}),
            json!({"cop":"=","args":[2_147_483_649_u64],"returns":["above31"]}),
        ],
        &[
            ("negative", 29, 5),
            ("out", 9_223_372_036_854_775_809, 64),
            ("above31", 2_147_483_649, 32),
        ],
    );
}
#[test]
fn reused_interpreter_and_shots() {
    let mut interpreter = PhirClassicalInterpreter::new();
    let mut ops = declarations();
    ops.extend([
        export(&["m"]),
        json!({"cop":"Result","args":["a"],"returns":["copy"]}),
    ]);
    interpreter.init(&program(ops), None).unwrap();
    interpreter.execute_program().unwrap();
    assert_interpreter(&interpreter, &[("m", 0, 2), ("copy", 0, 2)]);
    interpreter.shot_reinit();
    assert_interpreter(&interpreter, &[("m", 0, 2), ("copy", 0, 2)]);
    let mut native_ops = declarations();
    native_ops.extend([
        export(&["m"]),
        json!({"cop":"Result","args":["a"],"returns":["copy"]}),
    ]);
    let mut native = PhirJsonEngine::from_json(&program(native_ops)).unwrap();
    assert_native(&run(&mut native), &[("m", 0, 2), ("copy", 0, 2)]);
    ClassicalEngine::reset(&mut native).unwrap();
    assert_native(
        &native.get_results().unwrap(),
        &[("m", 0, 2), ("copy", 0, 2)],
    );
    interpreter.init(&program(declarations()), None).unwrap();
    assert_interpreter(&interpreter, &[("a", 0, 2), ("m", 0, 2), ("scratch", 0, 2)]);
}
fn rejection(directives: &[Value], bad: &str) {
    for nested in [false, true] {
        let mut ops = declarations();
        if nested {
            ops.push(json!({"block":"if","condition":{"cop":"==","args":[0,1]},"true_branch":directives}));
        } else {
            ops.extend_from_slice(directives);
        }
        let source = program(ops);
        let mut interpreter = PhirClassicalInterpreter::new();
        let errors = [
            PhirJsonEngine::from_json(&source).unwrap_err().to_string(),
            interpreter.init(&source, None).unwrap_err().to_string(),
            phir_json_to_module(&source).unwrap_err().to_string(),
        ];
        for error in errors {
            assert!(
                error.contains("cvar_export") && error.contains(bad),
                "{error}"
            );
        }
    }
}
#[test]
fn unknown_source() {
    rejection(&[export(&["missing"])], "missing");
}
#[test]
fn quantum_source() {
    rejection(&[export(&["q"])], "q");
}
#[test]
fn wrong_to_length() {
    rejection(
        &[json!({"data":"cvar_export","variables":["m"],"to":["out","extra"]})],
        "extra",
    );
}
#[test]
fn non_string_to() {
    rejection(
        &[json!({"data":"cvar_export","variables":["m"],"to":[42]})],
        "42",
    );
}
#[test]
fn invalid_to_shape() {
    rejection(
        &[json!({"data":"cvar_export","variables":["m"],"to":null})],
        "null",
    );
}
#[test]
fn conflicting_sources() {
    rejection(&[renamed("m", "out"), renamed("a", "out")], "out");
}
#[test]
fn declared_output() {
    rejection(&[renamed("m", "scratch")], "scratch");
}
#[test]
fn quantum_output() {
    rejection(&[renamed("m", "q")], "q");
}
#[test]
fn result_output() {
    rejection(
        &[
            renamed("m", "copy"),
            json!({"cop":"Result","args":["a"],"returns":[["copy",0]]}),
        ],
        "copy",
    );
}
#[test]
fn reserved_output() {
    rejection(&[renamed("m", "_width_out")], "_width_out");
}

#[test]
fn qparallel_export() {
    let mut ops = declarations();
    ops.push(json!({"block":"qparallel","ops":[export(&["m"])]}));
    check_classical(ops, &[("m", 0, 2)]);
}

#[test]
fn iterative_executor_collects_unexecuted_exports() {
    use pecos_phir_json::v0_1::block_iterative_executor::BlockIterativeExecutor;
    let mut ops = declarations();
    ops.push(
        json!({"block":"if","condition":{"cop":"==","args":[0,1]},"true_branch":[export(&["m"])]}),
    );
    let ast = PHIRProgram::from_json(&program(ops)).unwrap();
    let mut executor = BlockExecutor::new();
    BlockIterativeExecutor::new(&mut executor)
        .with_operations(&ast.ops)
        .process()
        .unwrap();
    assert_eq!(
        executor.process_export_mappings(),
        BTreeMap::from([("m".into(), 0)])
    );
}

#[test]
fn result_destination_exported_under_its_own_name() {
    for indexed in [false, true] {
        let mut ops = declarations();
        ops.extend([
            export(&["a"]),
            renamed("a", "alias"),
            json!({"qop":"X","args":[["q",0]]}),
            json!({"qop":"Measure","args":[["q",0]],"returns":[["m",0]]}),
            if indexed {
                json!({"cop":"Result","args":[["m",0]],"returns":[["a",1]]})
            } else {
                json!({"cop":"Result","args":["m"],"returns":["a"]})
            },
        ]);
        let source = program(ops);
        let value = if indexed { 2 } else { 1 };
        let entries = [("a", value, 2), ("alias", value, 2)];
        let mut native = PhirJsonEngine::from_json(&source).unwrap();
        let mut converted = PhirEngine::new(phir_json_to_module(&source).unwrap()).unwrap();
        let mut interpreter = PhirClassicalInterpreter::new();
        interpreter.init(&source, None).unwrap();
        for _ in 0..2 {
            assert_native(&run(&mut native), &entries);
            let shot = run(&mut converted);
            let (values, widths) = expected(&entries);
            assert_eq!(normalize(&shot, &widths), values);
            assert_eq!(shot.data.len(), entries.len());
            execute_interpreter(&mut interpreter);
            assert_interpreter(&interpreter, &entries);
            ClassicalEngine::reset(&mut native).unwrap();
            ClassicalEngine::reset(&mut converted).unwrap();
            interpreter.shot_reinit();
        }
    }
}

#[test]
fn own_export_survives_unexecuted_result() {
    let mut ops = declarations();
    ops.extend([
        renamed("a", "a"),
        json!({"block":"if","condition":{"cop":"==","args":[0,1]},"true_branch":[
            {"cop":"Result","args":["m"],"returns":["a"]}
        ]}),
    ]);
    check_classical(ops, &[("a", 0, 2)]);
}

#[test]
fn nested_declarations_without_exports_fail_at_load() {
    use pecos_phir_json::v0_1::{ast::Operation, block_iterative_executor::BlockIterativeExecutor};
    for data in ["cvar_define", "qvar_define"] {
        let declaration = if data == "cvar_define" {
            json!({"data":data,"variable":"inner","data_type":"u32","size":2})
        } else {
            json!({"data":data,"variable":"inner","size":1})
        };
        for block in [
            json!({"block":"sequence","ops":[declaration]}),
            json!({"block":"if","condition":{"cop":"==","args":[0,1]},"true_branch":[declaration]}),
            json!({"block":"if","condition":{"cop":"==","args":[1,1]},"true_branch":[],"false_branch":[
                {"block":"sequence","ops":[declaration]}
            ]}),
        ] {
            let ops: Vec<Operation> = vec![serde_json::from_value(block.clone()).unwrap()];
            let source = program(vec![block]);
            let ast = PHIRProgram {
                format: "PHIR/JSON".into(),
                version: "0.1.0".into(),
                metadata: BTreeMap::new(),
                ops: ops.clone(),
            };
            let mut interpreter = PhirClassicalInterpreter::new();
            let mut executor = BlockExecutor::new();
            let errors = [
                PhirJsonEngine::from_json(&source).unwrap_err().to_string(),
                PhirJsonEngine::from_program(ast).unwrap_err().to_string(),
                interpreter.init(&source, None).unwrap_err().to_string(),
                phir_json_to_module(&source).unwrap_err().to_string(),
                executor.execute_program(&ops).unwrap_err().to_string(),
                BlockIterativeExecutor::new(&mut executor)
                    .with_operations(&ops)
                    .process()
                    .unwrap_err()
                    .to_string(),
            ];
            for error in errors {
                assert!(error.contains(data) && error.contains("inner"), "{error}");
                assert!(error.contains("declarations inside blocks are only supported in programs that use cvar_export"), "{error}");
                assert!(
                    error.contains("move the declaration to the top level"),
                    "{error}"
                );
            }
        }
    }
}

#[test]
fn missing_and_null_export_fields_have_actionable_errors() {
    for (directive, field) in [
        (json!({"data":"cvar_export"}), "variables"),
        (json!({"data":"cvar_export","variables":null}), "variables"),
        (
            json!({"data":"cvar_export","variables":[null]}),
            "variables entry",
        ),
        (
            json!({"data":"cvar_export","variables":["m"],"to":[null]}),
            "to entry",
        ),
        (
            json!({"data":"cvar_export","variables":["m"],"to":null}),
            "to",
        ),
    ] {
        rejection(
            &[directive],
            &format!("invalid {field} <missing or null>; supply an array of strings"),
        );
    }
}

#[test]
fn conditional_result_is_static_and_resets() {
    for with_export in [false, true] {
        let mut ops = declarations();
        if with_export {
            ops.push(export(&["m"]));
        }
        ops.extend([
            json!({"qop":"H","args":[["q",0]]}),
            json!({"qop":"Measure","args":[["q",0]],"returns":[["m",0]]}),
            json!({"block":"if", "condition":{"cop":"==","args":[["m",0],1]},
                "true_branch":[{"cop":"Result","args":["m"],"returns":["copy"]}]}),
        ]);
        let source = program(ops);
        let mut engine = PhirJsonEngine::from_json(&source).unwrap();
        let mut interpreter = PhirClassicalInterpreter::new();
        interpreter.init(&source, None).unwrap();
        let mut keys = None;
        for shot in 0..128 {
            let outcome = u32::from(shot % 2 == 0);
            ClassicalEngine::reset(&mut engine).unwrap();
            assert!(!engine.generate_commands().unwrap().is_empty().unwrap());
            let mut outcomes = pecos_engines::ByteMessage::outcomes_builder();
            outcomes.add_outcomes(&[outcome as usize]);
            engine.handle_measurements(outcomes.build()).unwrap();
            assert!(engine.generate_commands().unwrap().is_empty().unwrap());
            let result = engine.get_results().unwrap();
            assert_eq!(integer(&result.data["copy"]), u64::from(outcome));
            assert_eq!(integer(&result.data["_width_copy"]), 2);
            let shot_keys: Vec<_> = result.data.keys().cloned().collect();
            if let Some(ref expected) = keys {
                assert_eq!(&shot_keys, expected);
            }
            keys = Some(shot_keys);
            interpreter.shot_reinit();
            let ops = interpreter.program_ops().to_vec();
            for op in &ops {
                interpreter
                    .execute_ops(std::slice::from_ref(op))
                    .next_batch()
                    .unwrap();
                if matches!(op, pecos_phir_json::v0_1::ast::Operation::QuantumOp { qop, .. } if qop == "Measure")
                {
                    let measured = BTreeMap::from([(("m".to_owned(), 0), i64::from(outcome))]);
                    interpreter
                        .receive_results(&[measured
                            .iter()
                            .map(|((name, bit), value)| (MeasKey::Bit(name.clone(), *bit), *value))
                            .collect()])
                        .unwrap();
                }
            }
            assert!(
                matches!(&interpreter.results(false).unwrap()["copy"], ResultValue::BitString(v) if v == &format!("{outcome:02b}"))
            );
        }
    }
}

#[test]
fn measure_requires_destinations_at_load() {
    for returns in [None, Some(json!([])), Some(json!([["m", 0], ["m", 1]]))] {
        let mut measure = json!({"qop":"Measure","args":[["q",0]]});
        if let Some(returns) = returns {
            measure["returns"] = returns;
        }
        let mut ops = declarations();
        ops.push(measure);
        let source = program(ops);
        for error in [
            PhirJsonEngine::from_json(&source)
                .err()
                .map(|e| e.to_string()),
            PhirClassicalInterpreter::new()
                .init(&source, None)
                .err()
                .map(|e| e.to_string()),
            phir_json_to_module(&source).err().map(|e| e.to_string()),
        ] {
            let error = error.expect("returnless/mismatched Measure must fail at load");
            assert!(
                error.contains("Measure")
                    && error.contains("give each measured qubit a destination"),
                "{error}"
            );
        }
    }
}

#[test]
fn private_measurement_bits_are_transferred() {
    let mut interpreter = PhirClassicalInterpreter::new();
    interpreter
        .init(
            &program(vec![
                json!({"data":"cvar_define","variable":"__x","data_type":"u32","size":1}),
            ]),
            None,
        )
        .unwrap();
    interpreter
        .receive_results(&[BTreeMap::from([(MeasKey::Bit("__x".into(), 0), 1)])])
        .unwrap();
    let bits = BTreeMap::from([(("__x".into(), 0), 1)]);
    assert_eq!(interpreter.result_bits(std::slice::from_ref(&bits)), bits);
}

#[test]
fn result_source_signedness_conflicts_fail_at_load() {
    for nested in [false, true] {
        let mut ops = vec![
            json!({"data":"cvar_define","variable":"a","data_type":"i32","size":2}),
            json!({"data":"cvar_define","variable":"b","data_type":"u32","size":3}),
        ];
        ops.push(json!({"cop":"Result","args":["a"],"returns":["copy"]}));
        ops.push(json!({"cop":"Result","args":["b"],"returns":["copy"]}));
        let flat = program(ops.clone());
        let copies = ops.split_off(2);
        ops.push(json!({"block":"if","condition":{"cop":"==","args":[0,1]},"true_branch":copies}));
        {
            let source = if nested { program(ops) } else { flat.clone() };
            assert!(
                PhirJsonEngine::from_json(&source)
                    .unwrap_err()
                    .to_string()
                    .contains("destination \"copy\" has conflicting signedness")
            );
            let error = PhirClassicalInterpreter::new()
                .init(&source, None)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("copy")
                    && error.contains("signedness")
                    && error.contains("\"a\"")
                    && error.contains("\"b\""),
                "{error}"
            );
        }
        let error = phir_json_to_module(&flat).unwrap_err().to_string();
        assert!(
            error.contains("copy")
                && error.contains("signedness")
                && error.contains("\"a\"")
                && error.contains("\"b\""),
            "{error}"
        );
    }
}

#[test]
fn result_storage_uses_declarations_and_final_values() {
    check_classical(
        vec![
            json!({"data":"cvar_define","variable":"source","data_type":"i32","size":4}),
            json!({"data":"cvar_define","variable":"declared","data_type":"i64","size":6}),
            export(&[]),
            json!({"block":"if","condition":{"cop":"==","args":[0,1]},"true_branch":[
                {"cop":"Result","args":["source","source"],"returns":["created","declared"]}
            ]}),
            json!({"cop":"=","args":[-2],"returns":["declared"]}),
        ],
        &[("created", 0, 5), ("declared", 126, 7)],
    );
}

#[test]
fn explicit_measurements_do_not_invent_or_overwrite_destinations() {
    let source = program(vec![
        json!({"data":"qvar_define","variable":"q","size":1}),
        json!({"data":"cvar_define","variable":"__x","data_type":"u32","size":1}),
        json!({"data":"cvar_define","variable":"measurement_0","data_type":"u32","size":1}),
        export(&["__x", "measurement_0"]),
        json!({"qop":"X","args":[["q",0]]}),
        json!({"qop":"Measure","args":[["q",0]],"returns":[["__x",0]]}),
    ]);
    assert_native(
        &run(&mut PhirJsonEngine::from_json(&source).unwrap()),
        &[("__x", 1, 1), ("measurement_0", 0, 1)],
    );
    let mut interpreter = PhirClassicalInterpreter::new();
    interpreter.init(&source, None).unwrap();
    execute_interpreter(&mut interpreter);
    assert_interpreter(&interpreter, &[("__x", 1, 1), ("measurement_0", 0, 1)]);
    let result = run(&mut PhirEngine::new(phir_json_to_module(&source).unwrap()).unwrap());
    assert_eq!(integer(&result.data["__x"]), 1);
    assert_eq!(integer(&result.data["measurement_0"]), 0);
}

#[test]
fn result_storage_precedes_copies_in_the_module_converter() {
    let source = program(vec![
        json!({"data":"qvar_define","variable":"q","size":1}),
        json!({"data":"cvar_define","variable":"source","data_type":"i32","size":4}),
        export(&[]),
        json!({"cop":"Result","args":["future"],"returns":["early"]}),
        json!({"qop":"X","args":[["q",0]]}),
        json!({"qop":"Measure","args":[["q",0]],"returns":[["source",3]]}),
        json!({"cop":"Result","args":["source","source"],"returns":["declared","future"]}),
        json!({"data":"cvar_define","variable":"declared","data_type":"u8","size":2}),
    ]);
    let entries = &[("declared", 0, 2), ("future", 8, 5), ("early", 0, 5)];
    assert_native(
        &run(&mut PhirJsonEngine::from_json(&source).unwrap()),
        entries,
    );
    let mut interpreter = PhirClassicalInterpreter::new();
    interpreter.init(&source, None).unwrap();
    execute_interpreter(&mut interpreter);
    assert_interpreter(&interpreter, entries);
    let result = run(&mut PhirEngine::new(phir_json_to_module(&source).unwrap()).unwrap());
    for (name, value, _) in entries {
        assert_eq!(integer(&result.data[*name]), *value);
    }
}

#[test]
fn result_widths_are_static_maxima() {
    for reverse in [false, true] {
        let mut copies = vec![
            json!({"cop":"Result","args":[["a",0]],"returns":[["out",0]]}),
            json!({"cop":"Result","args":[["b",1]],"returns":[["out",40]]}),
            json!({"cop":"Result","args":["a"],"returns":["whole"]}),
            json!({"cop":"Result","args":["b"],"returns":["whole"]}),
            json!({"cop":"Result","args":[["b",1]],"returns":["bit"]}),
            json!({"cop":"Result","args":["whole"],"returns":["chain"]}),
        ];
        if reverse {
            copies.reverse();
        }
        let mut ops = vec![
            json!({"data":"cvar_define","variable":"a","data_type":"u32","size":1}),
            json!({"data":"cvar_define","variable":"b","data_type":"u64","size":50}),
            export(&[]),
        ];
        ops.extend(copies);
        let source = program(ops);
        let expected = [
            ("out", 0, 41),
            ("whole", 0, 50),
            ("bit", 0, 1),
            ("chain", 0, 50),
        ];
        let mut native = PhirJsonEngine::from_json(&source).unwrap();
        assert_native(&run(&mut native), &expected);
        let mut classical = PhirClassicalInterpreter::new();
        classical.init(&source, None).unwrap();
        execute_interpreter(&mut classical);
        assert_interpreter(&classical, &expected);
        let mut module = PhirEngine::new(phir_json_to_module(&source).unwrap()).unwrap();
        let (values, widths) = crate::expected(&expected);
        assert_eq!(normalize(&run(&mut module), &widths), values);
    }
}

#[test]
fn declared_results_accept_all_source_types() {
    check_classical(
        vec![
            json!({"data":"cvar_define","variable":"a","data_type":"i32","size":2}),
            json!({"data":"cvar_define","variable":"b","data_type":"u32","size":3}),
            json!({"data":"cvar_define","variable":"out","data_type":"u32","size":8}),
            export(&[]),
            json!({"cop":"=","args":[-1],"returns":["a"]}),
            json!({"cop":"=","args":[5],"returns":["b"]}),
            json!({"cop":"Result","args":["a","b",["a",0]],"returns":["out","out",["out",6]]}),
        ],
        &[("out", 69, 8)],
    );
}

#[test]
fn result_created_names_are_hoisted() {
    check_classical(
        vec![
            json!({"data":"cvar_define","variable":"a","data_type":"u32","size":3}),
            export(&[]),
            json!({"cop":"=","args":["created"],"returns":["a"]}),
            json!({"cop":"=","args":[5],"returns":["created"]}),
            json!({"block":"if","condition":{"cop":"==","args":[0,1]},"true_branch":[
                {"cop":"Result","args":["a"],"returns":["created"]}
            ]}),
        ],
        &[("created", 5, 3)],
    );
}

#[test]
fn no_exports_report_all_classical_storage() {
    check_classical(
        vec![
            json!({"data":"qvar_define","variable":"q","size":1}),
            json!({"data":"cvar_define","variable":"measurement_user","data_type":"u32","size":2}),
            json!({"data":"cvar_define","variable":"a","data_type":"u32","size":2}),
            json!({"cop":"Result","args":["a"],"returns":["copy"]}),
        ],
        &[("a", 0, 2), ("measurement_user", 0, 2), ("copy", 0, 2)],
    );
    check_classical(
        vec![json!({"data":"qvar_define","variable":"q","size":1})],
        &[],
    );
}

#[test]
fn grouped_measure_counts_qubits() {
    let source = program(vec![
        json!({"data":"qvar_define","variable":"q","size":2}),
        json!({"data":"cvar_define","variable":"m","data_type":"u32","size":2}),
        json!({"qop":"Measure","args":[[["q",0],["q",1]]],"returns":[["m",0],["m",1]]}),
    ]);
    PhirJsonEngine::from_json(&source).unwrap();
    PhirClassicalInterpreter::new().init(&source, None).unwrap();
    phir_json_to_module(&source).unwrap();
}

#[test]
fn malformed_measure_diagnostics_name_the_operation() {
    for (args, returns) in [
        (Value::Null, json!([["m", 0]])),
        (json!("q"), json!([["m", 0]])),
        (json!({"q":0}), json!([["m", 0]])),
        (json!([["q", 0]]), json!("m")),
        (json!([["q", 0]]), json!({"m":0})),
        (json!([[["q", 0], ["q", 1]]]), json!([["m", 0]])),
    ] {
        let source = program(vec![json!({"qop":"Measure","args":args,"returns":returns})]);
        let diagnostic = format!(
            "Measure args {args}: invalid args or returns {returns}; supply arrays and give each measured qubit a destination"
        );
        for error in [
            PhirJsonEngine::from_json(&source).unwrap_err().to_string(),
            PhirClassicalInterpreter::new()
                .init(&source, None)
                .unwrap_err()
                .to_string(),
            phir_json_to_module(&source).unwrap_err().to_string(),
        ] {
            assert!(error.contains(&diagnostic), "{error}");
        }
    }
}

#[test]
fn indexed_results_reserve_unexecuted_writes() {
    check_classical(
        vec![
            json!({"data":"cvar_define","variable":"s","data_type":"i32","size":2}),
            json!({"data":"cvar_define","variable":"u","data_type":"u32","size":5}),
            export(&[]),
            json!({"cop":"=","args":[-1],"returns":["s"]}),
            json!({"cop":"Result","args":[["s",0],["s",0]],"returns":[["out",0],"bit"]}),
            json!({"block":"if","condition":{"cop":"==","args":[0,1]},"true_branch":[
                {"cop":"Result","args":[["u",0],["u",0]],"returns":[["out",40],"bit"]}
            ]}),
        ],
        &[("out", 1, 41), ("bit", 1, 1)],
    );
}
