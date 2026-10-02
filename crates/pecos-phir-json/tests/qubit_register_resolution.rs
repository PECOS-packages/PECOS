use pecos_core::Gate;
use pecos_engines::byte_message::builder::ByteMessageBuilder;
use pecos_phir::{Module, builtin_ops::BuiltinOp, ops::Operation};
use pecos_phir_json::v0_1::{
    ast::{Operation as JsonOperation, PHIRProgram, QubitArg, infer_size},
    classical_interpreter::{PhirClassicalInterpreter, QOpArgs},
    operations::OperationProcessor,
};
use serde_json::{Value, json};

use pecos_phir_json::phir_json_to_module;

#[test]
fn issue_819_reproduction() {
    let json = r#"{"format":"PHIR/JSON","version":"0.1.0","metadata":{"num_qubits":4},"ops":[
      {"data":"qvar_define","data_type":"qubits","variable":"a","size":2},
      {"data":"qvar_define","data_type":"qubits","variable":"b","size":2},
      {"qop":"X","args":[["b",0]],"returns":[]},
      {"qop":"X","args":[["a",1]],"returns":[]},
      {"qop":"CX","args":[[["b",0],["a",1]]],"returns":[]}
    ]}"#;
    let module = phir_json_to_module(json).unwrap();
    let operands: Vec<Vec<u32>> = module.body.blocks[0]
        .operations
        .iter()
        .filter(|instruction| matches!(instruction.operation, Operation::Quantum(_)))
        .map(|instruction| instruction.operands.iter().map(|value| value.id).collect())
        .collect();
    assert_eq!(operands, [vec![2], vec![1], vec![2, 1]]);
    assert_ne!(operands[0], operands[1]);
}

fn program(registers: &[(&str, usize)], gates: Vec<Value>) -> Value {
    let mut ops: Vec<_> = registers
        .iter()
        .map(|(name, size)| {
            json!({"data":"qvar_define", "data_type":"qubits", "variable":name, "size":size})
        })
        .collect();
    ops.extend(gates);
    json!({"format":"PHIR/JSON", "version":"0.1.0", "ops":ops})
}

// Expected flat IDs are explicit oracle inputs, never derived from the resolver.
fn equivalent_programs() -> Vec<(Value, Value)> {
    vec![
        (
            program(
                &[("a", 2), ("b", 2)],
                vec![
                    json!({"qop":"X", "args":[["b",0]], "returns":[]}),
                    json!({"qop":"X", "args":[["a",1]], "returns":[]}),
                    json!({"qop":"H", "args":[["b",1]], "returns":[]}),
                    json!({"qop":"CX", "args":[[["b",0],["a",1]]], "returns":[]}),
                    json!({"qop":"Measure", "args":[["b",1]], "returns":[]}),
                ],
            ),
            program(
                &[("q", 4)],
                vec![
                    json!({"qop":"X", "args":[["q",2]], "returns":[]}),
                    json!({"qop":"X", "args":[["q",1]], "returns":[]}),
                    json!({"qop":"H", "args":[["q",3]], "returns":[]}),
                    json!({"qop":"CX", "args":[[["q",2],["q",1]]], "returns":[]}),
                    json!({"qop":"Measure", "args":[["q",3]], "returns":[]}),
                ],
            ),
        ),
        (
            program(
                &[("x", 1), ("y", 3), ("z", 2)],
                vec![
                    json!({"qop":"X", "args":[["y",2]], "returns":[]}),
                    json!({"qop":"H", "args":[["z",1]], "returns":[]}),
                    json!({"qop":"CX", "args":[[["z",0],["x",0]],[["y",0],["z",1]]], "returns":[]}),
                    json!({"qop":"Measure", "args":[["z",0],["y",1]], "returns":[]}),
                ],
            ),
            program(
                &[("q", 6)],
                vec![
                    json!({"qop":"X", "args":[["q",3]], "returns":[]}),
                    json!({"qop":"H", "args":[["q",5]], "returns":[]}),
                    json!({"qop":"CX", "args":[[["q",4],["q",0]],[["q",1],["q",5]]], "returns":[]}),
                    json!({"qop":"Measure", "args":[["q",4],["q",2]], "returns":[]}),
                ],
            ),
        ),
        // Declaration order must not become alphabetical order.
        (
            program(
                &[("z", 2), ("a", 1)],
                vec![json!({"qop":"CX", "args":[["a",0],["z",1]], "returns":[]})],
            ),
            program(
                &[("q", 3)],
                vec![json!({"qop":"CX", "args":[["q",2],["q",1]], "returns":[]})],
            ),
        ),
    ]
}

fn executable_module(program: &Value) -> Module {
    let mut module = phir_json_to_module(&program.to_string()).unwrap();
    // The declarations necessarily differ. Compare the entire remaining module,
    // including quantum instructions, operands, results, types, and attributes.
    module.body.blocks[0]
        .operations
        .retain(|op| !matches!(op.operation, Operation::Builtin(BuiltinOp::VarDefine(_))));
    module
}

fn processor(program: &Value) -> (OperationProcessor, PHIRProgram) {
    let program: PHIRProgram = serde_json::from_value(program.clone()).unwrap();
    let mut processor = OperationProcessor::new();
    for op in &program.ops {
        if let JsonOperation::VariableDefinition {
            data,
            data_type,
            variable,
            size,
        } = op
        {
            processor
                .handle_variable_definition(data, data_type, variable, infer_size(data_type, *size))
                .unwrap();
        }
    }
    (processor, program)
}

fn commands(program: &Value) -> Vec<Gate> {
    let (processor, program) = processor(program);
    let mut builder = ByteMessageBuilder::new();
    let _ = builder.for_quantum_operations();
    for op in &program.ops {
        if let JsonOperation::QuantumOp {
            qop, args, angles, ..
        } = op
        {
            let (name, ids, angles) = processor
                .process_quantum_op(qop, angles.as_ref(), args)
                .unwrap();
            processor
                .add_quantum_operation_to_builder(&mut builder, &name, &ids, &angles)
                .unwrap();
        }
    }
    builder.build().quantum_ops().unwrap()
}

#[test]
fn module_equivalence_oracle() {
    for (registers, flat) in equivalent_programs() {
        assert_eq!(executable_module(&registers), executable_module(&flat));
    }
}

#[test]
fn collector_equivalence_oracle() {
    for (registers, flat) in equivalent_programs() {
        assert_eq!(commands(&registers), commands(&flat));
    }
    let (registers, _) = equivalent_programs().remove(0);
    let ids: Vec<Vec<usize>> = commands(&registers)
        .iter()
        .map(|gate| gate.qubits.iter().map(|q| q.0).collect())
        .collect();
    assert_eq!(ids, [vec![2], vec![1], vec![3], vec![2, 1], vec![3]]);
}

#[test]
fn unknown_and_out_of_bounds_qubits_are_rejected() {
    for (register, index, error) in [("missing", 0, "missing"), ("a", 2, "out of bounds")] {
        for args in [
            json!([[register, index]]),
            json!([[["b", 0], [register, index]]]),
        ] {
            let input = program(&[("a", 2), ("b", 2)], vec![json!({"qop":"X", "args":args})]);
            let err = phir_json_to_module(&input.to_string()).unwrap_err();
            assert!(err.to_string().contains(error), "{err}");
            let (processor, _) = processor(&input);
            let args: Vec<QubitArg> = serde_json::from_value(args).unwrap();
            let err = processor.process_quantum_op("X", None, &args).unwrap_err();
            assert!(err.to_string().contains(error), "{err}");
            let mut interpreter = PhirClassicalInterpreter::new();
            interpreter.init(&input.to_string(), None).unwrap();
            let err = interpreter
                .make_qop("X", &None, &args, &[], &None)
                .unwrap_err();
            assert!(err.to_string().contains(error), "{err}");
        }
    }
}

#[test]
fn quantum_returns_resolve_but_measurement_returns_remain_classical() {
    let input = program(
        &[("a", 2), ("b", 2)],
        vec![
            json!({"data":"cvar_define", "data_type":"u32", "variable":"m", "size":2}),
            json!({"qop":"X", "args":[["b",0]], "returns":[["b",0]]}),
            json!({"qop":"Measure", "args":[["b",1]], "returns":[["m",1]]}),
        ],
    );
    let module = phir_json_to_module(&input.to_string()).unwrap();
    let gates: Vec<_> = module.body.blocks[0]
        .operations
        .iter()
        .filter(|op| matches!(op.operation, Operation::Quantum(_)))
        .collect();
    assert_eq!(gates[0].results[0].id, 2);
    assert_eq!(gates[1].operands[0].id, 3);
    assert!(
        gates[1].results[0].id >= 4,
        "classical results must not alias qubits"
    );
    let (mut processor, _) = processor(&input);
    processor
        .record_measurement_returns(1, &[("m".into(), 1)])
        .unwrap();
    processor.handle_measurements(&[1], &[]).unwrap();
    assert_eq!(processor.environment.get_raw("m"), Some(2));

    for target in [json!(["missing", 0]), json!(["a", 2])] {
        let input = program(
            &[("a", 2)],
            vec![json!({"qop":"X", "args":[["a",0]], "returns":[target]})],
        );
        assert!(phir_json_to_module(&input.to_string()).is_err());
    }
}

#[test]
fn machine_commands_and_barriers_validate_registers() {
    let input = program(&[("a", 2), ("b", 2)], vec![]);
    let (processor, _) = processor(&input);
    let args = vec![
        QubitArg::SingleQubit(("b".into(), 0)),
        QubitArg::MultipleQubits(vec![("a".into(), 1), ("b".into(), 1)]),
    ];
    for name in ["Idle", "Delay", "Transport", "Timing"] {
        let result = processor
            .process_machine_op(name, Some(&args), Some(&(1.0, "ns".into())), None)
            .unwrap();
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();
        processor
            .add_machine_operation_to_builder(&mut builder, &result)
            .unwrap();
        let gates = builder.build().quantum_ops().unwrap();
        if name == "Timing" {
            assert_eq!(gates, []);
        } else {
            assert_eq!(
                gates[0].qubits.iter().map(|q| q.0).collect::<Vec<_>>(),
                [2, 1, 3]
            );
        }
        for bad in [("missing".into(), 0), ("a".into(), 2)] {
            assert!(
                processor
                    .process_machine_op(
                        name,
                        Some(&vec![QubitArg::SingleQubit(bad.clone())]),
                        None,
                        None
                    )
                    .is_err()
            );
            assert!(
                processor
                    .process_meta_instruction("barrier", &[bad])
                    .is_err()
            );
        }
    }
    let barrier = processor
        .process_meta_instruction("barrier", &[("b".into(), 0), ("a".into(), 1)])
        .unwrap();
    let mut builder = ByteMessageBuilder::new();
    let _ = builder.for_quantum_operations();
    processor
        .add_meta_instruction_to_builder(&mut builder, &barrier)
        .unwrap();
    assert_eq!(builder.build().quantum_ops().unwrap(), []);
}

#[test]
fn interpreter_uses_global_ids_and_preserves_classical_namespace() {
    let input = program(
        &[("z", 1), ("y", 3), ("x", 2)],
        vec![json!({"data":"cvar_define", "data_type":"u32", "variable":"x", "size":2})],
    );
    let mut interpreter = PhirClassicalInterpreter::new();
    assert_eq!(interpreter.init(&input.to_string(), None).unwrap(), 6);
    let args = vec![QubitArg::MultipleQubits(vec![
        ("x".into(), 1),
        ("y".into(), 2),
    ])];
    for _ in 0..2 {
        let op = interpreter
            .make_qop("CX", &None, &args, &[], &None)
            .unwrap();
        assert!(matches!(op.args, QOpArgs::Multi(ids) if ids == vec![vec![5, 3]]));
        let op = interpreter
            .make_mop("Idle", &Some(args.clone()), &None, &None)
            .unwrap();
        assert!(matches!(op.args, Some(QOpArgs::Multi(ids)) if ids == vec![vec![5, 3]]));
        interpreter.shot_reinit();
    }
}

#[test]
fn offsets_survive_redefinitions_clone_and_reset() {
    let mut input = program(&[("z", 1), ("a", 3), ("b", 2)], vec![]);
    input["ops"].as_array_mut().unwrap().insert(
        1,
        json!({"data":"cvar_define", "data_type":"u32", "variable":"c", "size":7}),
    );
    let (mut processor, _) = processor(&input);
    processor.add_quantum_variable("a", 3).unwrap();
    processor.add_classical_variable("m", "u32", 3).unwrap();
    assert!(processor.add_quantum_variable("a", 4).is_err());
    let mut environment = processor.environment.clone();
    environment.reset_values();
    assert_eq!(environment.count_qubits(), 6);
    assert_eq!(environment.resolve_qubit("b", 1).unwrap(), 5);
    assert!(environment.resolve_qubit("m", 0).is_err());
}

fn assert_ssa_exhausted(input: &Value) {
    let result = std::panic::catch_unwind(|| phir_json_to_module(&input.to_string()));
    let error = result
        .expect("SSA exhaustion must return an error, not panic")
        .unwrap_err();
    assert!(
        error.to_string().contains("SSA ID space exhausted"),
        "{error}"
    );
}

#[test]
fn ssa_exhaustion_at_declaration_emission() {
    let max = usize::try_from(u32::MAX).unwrap();
    assert_ssa_exhausted(&program(&[("q", max)], vec![]));
    assert_ssa_exhausted(&program(
        &[("q", max - 1)],
        vec![json!({"data":"cvar_define", "data_type":"u32", "variable":"m", "size":1})],
    ));
    // The reservation itself fits: one declaration can still be emitted here.
    assert!(phir_json_to_module(&program(&[("q", max - 1)], vec![]).to_string()).is_ok());
}

#[test]
fn ssa_exhaustion_at_instruction_emission() {
    let max = usize::try_from(u32::MAX).unwrap();
    for op in [
        json!({"qop":"X", "args":[["q",0]]}),
        json!({"qop":"X", "args":[["q",0]], "returns":["result"]}),
        json!({"qop":"Measure", "args":[["q",0]], "returns":[["m",0]]}),
        json!({"cop":"Result", "args":["source"], "returns":[]}),
        json!({"cop":"Result", "args":[], "returns":["destination"]}),
    ] {
        // The qvar declaration consumes the last available allocation.
        assert_ssa_exhausted(&program(&[("q", max - 1)], vec![op]));
    }
}

#[test]
fn ssa_exhaustion_at_measurement_combining_emission() {
    let max = usize::try_from(u32::MAX).unwrap();
    let single = vec![
        json!({"data":"cvar_define", "data_type":"u32", "variable":"m", "size":2}),
        json!({"qop":"Measure", "args":[["q",0]], "returns":[["m",0]]}),
    ];
    // Two declarations and one measurement fit, but the combining zero constant does not.
    assert_ssa_exhausted(&program(&[("q", max - 3)], single));

    let multiple = vec![
        json!({"data":"cvar_define", "data_type":"u32", "variable":"m", "size":2}),
        json!({"qop":"Measure", "args":[["q",0],["q",1]], "returns":[["m",0],["m",1]]}),
    ];
    // After two declarations and two measurement results, exhaust at each
    // generated constant, cast, shift, and OR (including the bit-zero branch).
    for available in 4..10 {
        assert_ssa_exhausted(&program(&[("q", max - available)], multiple.clone()));
    }
    assert!(phir_json_to_module(&program(&[("q", max - 10)], multiple).to_string()).is_ok());
}

#[test]
fn converter_rejects_non_qubit_quantum_declarations() {
    let mut input = program(&[("q", 1)], vec![]);
    for data_type in [json!("u32"), json!("unknown"), json!(null), json!(42)] {
        input["ops"][0]["data_type"] = data_type;
        let error = phir_json_to_module(&input.to_string()).unwrap_err();
        assert!(
            error.to_string().contains("data_type") && error.to_string().contains("'q'"),
            "{error}"
        );
    }
}

#[test]
fn omitted_quantum_data_type_defaults_to_qubits() {
    let explicit = program(
        &[("z", 1), ("a", 3), ("b", 2)],
        vec![
            json!({"qop":"X", "args":[["a",2]], "returns":[]}),
            json!({"qop":"X", "args":[["b",1]], "returns":[]}),
            json!({"qop":"CX", "args":[[["b",0],["z",0]]], "returns":[]}),
        ],
    );
    let mut omitted = explicit.clone();
    for index in [1, 2] {
        omitted["ops"][index]
            .as_object_mut()
            .unwrap()
            .remove("data_type");
    }
    let module = phir_json_to_module(&omitted.to_string()).unwrap();
    // Also compare declarations: omitted types must emit VarDefine("qubits"),
    // not just pass validation and resolve operands correctly.
    assert_eq!(module, phir_json_to_module(&explicit.to_string()).unwrap());
    let operands: Vec<Vec<u32>> = module.body.blocks[0]
        .operations
        .iter()
        .filter(|instruction| matches!(instruction.operation, Operation::Quantum(_)))
        .map(|instruction| instruction.operands.iter().map(|value| value.id).collect())
        .collect();
    assert_eq!(operands, [vec![3], vec![5], vec![4, 0]]);
}

#[test]
fn converter_requires_quantum_variable_and_size() {
    for (field, error) in [
        ("variable", "requires a variable name"),
        ("size", "requires a size"),
    ] {
        let mut input = program(&[("q", 1)], vec![]);
        let declaration = input["ops"][0].as_object_mut().unwrap();
        declaration.remove("data_type");
        declaration.remove(field);
        let result = phir_json_to_module(&input.to_string()).unwrap_err();
        assert!(result.to_string().contains(error), "{result}");
    }
}

#[test]
fn quantum_declaration_compatibility_differences_are_explicit() {
    let duplicate = program(&[("q", 1), ("q", 1)], vec![]);
    assert!(
        phir_json_to_module(&duplicate.to_string())
            .unwrap_err()
            .to_string()
            .contains("already exists")
    );
    let mut interpreter = PhirClassicalInterpreter::new();
    assert!(
        interpreter
            .init(&duplicate.to_string(), None)
            .unwrap_err()
            .to_string()
            .contains("already exists")
    );
    let (processor, _) = processor(&duplicate);
    assert_eq!(processor.environment.count_qubits(), 1);

    let mut missing_size = program(&[("q", 0)], vec![]);
    missing_size["ops"][0]
        .as_object_mut()
        .unwrap()
        .remove("size");
    assert!(
        phir_json_to_module(&missing_size.to_string())
            .unwrap_err()
            .to_string()
            .contains("requires a size")
    );
    assert!(
        interpreter
            .init(&missing_size.to_string(), None)
            .unwrap_err()
            .to_string()
            .contains("requires a size")
    );
}
