use pecos_engines::{ClassicalEngine, byte_message::ByteMessage};
use pecos_phir::ops::Operation as PhirOperation;
use pecos_phir_json::{
    phir_json_to_module,
    v0_1::{
        ast::{Operation, PHIRProgram, QubitArg, validate_quantum_declaration},
        block_executor::BlockExecutor,
        classical_interpreter::{PhirClassicalInterpreter, QOpArgs},
        engine::PhirJsonEngine,
        operations::OperationProcessor,
    },
};
use serde_json::{Value, json};

fn program(ops: &[Value]) -> String {
    json!({"format":"PHIR/JSON","version":"0.1.0","ops":ops}).to_string()
}

#[test]
fn declaration_verdicts() {
    // The first and third rows reproduced opposite converter/AST verdicts at bb5932b06.
    for (label, declaration, accepted, error_text) in [
        (
            "omitted type",
            json!({"data":"qvar_define","variable":"q","size":2}),
            true,
            "",
        ),
        (
            "explicit qubits",
            json!({"data":"qvar_define","data_type":"qubits","variable":"q","size":2}),
            true,
            "",
        ),
        (
            "wrong type",
            json!({"data":"qvar_define","data_type":"u32","variable":"q","size":2}),
            false,
            "u32",
        ),
        (
            "missing size",
            json!({"data":"qvar_define","data_type":"qubits","variable":"q"}),
            false,
            "size",
        ),
        (
            "missing type and size",
            json!({"data":"qvar_define","variable":"q"}),
            false,
            "size",
        ),
        (
            "classical missing type",
            json!({"data":"cvar_define","variable":"q","size":2}),
            false,
            "data_type",
        ),
        (
            "null type",
            json!({"data":"qvar_define","data_type":null,"variable":"q","size":2}),
            false,
            "data_type",
        ),
        (
            "boolean size",
            json!({"data":"qvar_define","variable":"q","size":true}),
            false,
            "size",
        ),
        (
            "fractional size",
            json!({"data":"qvar_define","variable":"q","size":2.0}),
            false,
            "size",
        ),
        (
            "negative size",
            json!({"data":"qvar_define","variable":"q","size":-1}),
            false,
            "size",
        ),
        (
            "export field cannot bypass declaration validation",
            json!({"data":"qvar_define","data_type":"u32","variable":"q","size":2,"variables":[]}),
            false,
            "u32",
        ),
        (
            "zero size",
            json!({"data":"qvar_define","variable":"q","size":0}),
            false,
            "positive size",
        ),
    ] {
        let input = program(std::slice::from_ref(&declaration));
        let mut interpreter = PhirClassicalInterpreter::new();
        let engine = PhirJsonEngine::from_json(&input);
        let verdicts = [
            (
                "converter",
                phir_json_to_module(&input)
                    .map(|_| ())
                    .map_err(|e| e.to_string()),
            ),
            (
                "AST",
                serde_json::from_str::<PHIRProgram>(&input)
                    .map(|_| ())
                    .map_err(|e| e.to_string()),
            ),
            (
                "interpreter",
                interpreter
                    .init(&input, None)
                    .map(|_| ())
                    .map_err(|e| e.to_string()),
            ),
            (
                "engine",
                engine.as_ref().map(|_| ()).map_err(ToString::to_string),
            ),
        ];
        for (entry, result) in verdicts {
            assert_eq!(
                result.is_ok(),
                accepted,
                "{label}: {entry}: {result:?}; verdict must match across entry points"
            );
            if let Err(error) = result {
                assert!(error.contains(error_text), "{label}: {entry}: {error}");
                if matches!(label, "wrong type" | "missing size" | "zero size") {
                    assert!(error.contains("'q'"), "{entry}: {error}");
                }
            }
        }
        if accepted {
            let expected = declaration["size"].as_u64().unwrap();
            assert_eq!(interpreter.num_qubits(), usize::try_from(expected).unwrap());
            let engine = engine.unwrap();
            assert_eq!(engine.num_qubits(), usize::try_from(expected).unwrap());
            assert_eq!(
                engine.processor.environment.resolve_qubit("q", 0).unwrap(),
                0
            );
            let ast: PHIRProgram = serde_json::from_str(&input).unwrap();
            assert!(
                matches!(&ast.ops[0], Operation::VariableDefinition { data_type, size: Some(size), .. }
                if data_type == "qubits" && *size == usize::try_from(expected).unwrap())
            );
        }
    }
}

#[test]
fn optional_type_preserves_declaration_order_ids() {
    for explicit_type in [false, true] {
        let mut q = json!({"data":"qvar_define","variable":"a","size":2});
        if explicit_type {
            q["data_type"] = json!("qubits");
        }
        let input = program(&[
            json!({"data":"qvar_define","variable":"z","size":2}),
            q,
            json!({"qop":"CX","args":[["a",1],["z",1]]}),
        ]);
        let module = phir_json_to_module(&input).unwrap();
        let gate = module.body.blocks[0]
            .operations
            .iter()
            .find(|op| matches!(op.operation, PhirOperation::Quantum(_)))
            .unwrap();
        assert_eq!(
            gate.operands.iter().map(|v| v.id).collect::<Vec<_>>(),
            [3, 1]
        );
        let mut engine = PhirJsonEngine::from_json(&input).unwrap();
        assert_eq!(
            engine.num_qubits(),
            4,
            "header must register both quantum declarations"
        );
        assert_eq!(
            engine.processor.environment.resolve_qubit("a", 1).unwrap(),
            3
        );
        assert_eq!(
            engine.processor.environment.resolve_qubit("z", 1).unwrap(),
            1
        );
        assert_eq!(command_qubits(&engine.generate_commands().unwrap()), [3, 1]);
        let ast: PHIRProgram = serde_json::from_str(&input).unwrap();
        let mut executor = BlockExecutor::new();
        for op in &ast.ops[..2] {
            executor.process_operation(op).unwrap();
        }
        let args = vec![
            QubitArg::SingleQubit(("a".into(), 1)),
            QubitArg::SingleQubit(("z".into(), 1)),
        ];
        let (_, ids, _) = executor
            .processor
            .process_quantum_op("X", None, &args)
            .unwrap();
        assert_eq!(ids, [3, 1]);
        let mut interpreter = PhirClassicalInterpreter::new();
        assert_eq!(interpreter.init(&input, None).unwrap(), 4);
        let op = interpreter.make_qop("X", &None, &args, &[], &None).unwrap();
        assert!(matches!(op.args, QOpArgs::Single(ids) if ids == [3, 1]));
    }
}

#[test]
fn processor_duplicate_policy() {
    let mut processor = OperationProcessor::new();
    processor
        .handle_variable_definition("qvar_define", "qubits", "q", 2)
        .unwrap();
    processor
        .handle_variable_definition("qvar_define", "qubits", "q", 2)
        .expect_err("identical quantum redeclaration must be rejected");
    assert_eq!(processor.environment.count_qubits(), 2);
    let error = processor
        .handle_variable_definition("qvar_define", "qubits", "q", 3)
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("Variable 'q' is already declared as quantum; cannot redeclare as quantum")
    );
    assert_eq!(processor.environment.count_qubits(), 2);
}

#[test]
fn converter_and_interpreter_reject_all_duplicates() {
    for second_size in [2, 3] {
        let input = program(&[
            json!({"data":"qvar_define","variable":"q","size":2}),
            json!({"data":"qvar_define","variable":"q","size":second_size}),
        ]);
        assert!(phir_json_to_module(&input).is_err());
        assert!(PhirClassicalInterpreter::new().init(&input, None).is_err());
    }
}

#[test]
fn constructed_ast_cannot_bypass_validation() {
    for (data_type, size) in [("u32", Some(2)), ("qubits", None), ("qubits", Some(0))] {
        let op = Operation::VariableDefinition {
            data: "qvar_define".into(),
            data_type: data_type.into(),
            variable: "q".into(),
            size,
        };
        let mut executor = BlockExecutor::new();
        assert!(executor.process_operation(&op).is_err());
        assert_eq!(executor.processor.environment.count_qubits(), 0);
        let program = PHIRProgram {
            format: "PHIR/JSON".into(),
            version: "0.1.0".into(),
            metadata: std::collections::BTreeMap::default(),
            ops: vec![op],
        };
        assert!(PhirJsonEngine::from_program(program).is_err());
    }
    let mut processor = OperationProcessor::new();
    let error = processor
        .handle_variable_definition("qvar_define", "u32", "q", 2)
        .unwrap_err();
    assert!(error.to_string().contains("'q'"));
    assert!(error.to_string().contains("u32"));
}

#[test]
fn quantum_size_conversion_checks_platform_overflow() {
    let overflow = (usize::MAX as u128) + 1;
    assert!(
        validate_quantum_declaration("q", None, Some(overflow))
            .unwrap_err()
            .to_string()
            .contains("too large")
    );
    assert_eq!(
        validate_quantum_declaration("q", None, Some(usize::MAX)).unwrap(),
        ("q", usize::MAX)
    );
}

fn command_qubits(message: &ByteMessage) -> Vec<usize> {
    message
        .quantum_ops()
        .unwrap()
        .iter()
        .flat_map(|gate| gate.qubits.iter().map(|qubit| qubit.0))
        .collect()
}

#[test]
fn processor_rejects_zero_size() {
    let mut processor = OperationProcessor::new();
    let result = processor.handle_variable_definition("qvar_define", "qubits", "q", 0);
    assert!(
        result.is_err(),
        "processor must reject zero size, got {result:?}"
    );
    let message = result.unwrap_err().to_string();
    assert!(
        message.contains("'q'") && message.contains("positive size"),
        "{message}"
    );
    assert_eq!(processor.environment.count_qubits(), 0);
    assert!(!processor.environment.has_variable("q"));
}

#[test]
fn declarations_execute_through_blocks_and_engine_commands() {
    let input = program(&[
        json!({"data":"qvar_define","variable":"z","size":2}),
        json!({"data":"qvar_define","variable":"a","size":2}),
        json!({"qop":"CX","args":[["a",1],["z",1]]}),
    ]);
    let ast: PHIRProgram = serde_json::from_str(&input).unwrap();
    let mut executor = BlockExecutor::new();
    executor.execute_program(&ast.ops).unwrap();
    assert_eq!(executor.processor.environment.count_qubits(), 4);
    assert_eq!(command_qubits(&executor.get_builder().build()), [3, 1]);

    let mut engine = PhirJsonEngine::from_json(&input).unwrap();
    // Runtime execution uses the declarations registered during construction.
    let commands = engine
        .generate_commands()
        .expect("registered qubits must be available while executing gates");
    assert_eq!(engine.num_qubits(), 4);
    assert_eq!(command_qubits(&commands), [3, 1]);

    let mut ast: PHIRProgram = serde_json::from_str(&input).unwrap();
    ast.ops.push(Operation::VariableDefinition {
        data: "qvar_define".into(),
        data_type: "qubits".into(),
        variable: "z".into(),
        size: Some(3),
    });
    let error = PhirJsonEngine::from_program(ast)
        .map(|_| ())
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("Variable 'z' is already declared as quantum; cannot redeclare as quantum"),
        "{error}"
    );
}

#[test]
fn empty_quantum_circuit_round_trip() {
    // The Python producer test checks its entire output against this same fixture.
    let input = include_str!("fixtures/empty_quantum_circuit.phir.json");
    let module = phir_json_to_module(input).unwrap();
    assert_eq!(module.body.blocks[0].operations, []);
    let ast: PHIRProgram = serde_json::from_str(input).unwrap();
    assert!(ast.ops.is_empty());
    let mut interpreter = PhirClassicalInterpreter::new();
    assert_eq!(interpreter.init(input, None).unwrap(), 0);
    assert!(interpreter.execute_program().unwrap().is_empty());
    let mut engine = PhirJsonEngine::from_json(input).unwrap();
    assert_eq!(engine.num_qubits(), 0);
    let qubits = command_qubits(&engine.generate_commands().unwrap());
    assert!(
        qubits.is_empty(),
        "expected no command qubits, got {qubits:?}"
    );
    let engine = PhirJsonEngine::from_program(ast).unwrap();
    assert_eq!(engine.num_qubits(), 0);
}
