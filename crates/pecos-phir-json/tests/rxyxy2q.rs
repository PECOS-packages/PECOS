use pecos_core::{Angle64, Gate};
use pecos_engines::ClassicalEngine;
use pecos_phir::PhirEngine;
use pecos_phir_json::v0_1::ast::{Operation, PHIRProgram};
use pecos_phir_json::v0_1::operations::OperationProcessor;
use pecos_phir_json::{PhirJsonEngine, phir_json_to_module};
use serde_json::{Value, json};

fn program(angles: Value, args: Value) -> Value {
    let mut program = json!({"format": "PHIR/JSON", "version": "0.1.0", "ops": [
        {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 3},
        {"qop": "RXYXY2Q"}
    ]});
    program["ops"][1]["angles"] = angles;
    program["ops"][1]["args"] = args;
    program
}

// ByteMessage stores f64 radians, not raw Angle64 fractions. Its documented
// round trip can lose low fraction bits even when the source f64 is unchanged.
fn message_angle(radians: f64) -> Angle64 {
    let angle = Angle64::try_from_radians(radians).unwrap();
    Angle64::try_from_radians(angle.to_radians()).unwrap()
}

#[test]
fn rxyxy2q_json_units_and_execution() {
    let theta = -0.73_f64;
    let phi = 0.41_f64;
    for (unit, values) in [
        ("rad", [theta, phi]),
        ("deg", [theta.to_degrees(), phi.to_degrees()]),
        (
            "pi",
            [theta / std::f64::consts::PI, phi / std::f64::consts::PI],
        ),
    ] {
        let json = program(json!([values, unit]), json!([[["q", 2], ["q", 0]]]));
        let source = json.to_string();
        let ast: PHIRProgram = serde_json::from_str(&source).unwrap();
        let Operation::QuantumOp {
            qop,
            angles: Some(angles),
            args,
            ..
        } = &ast.ops[1]
        else {
            panic!("expected rotation")
        };
        assert_eq!(qop, "RXYXY2Q");
        assert!((angles[0] - theta).abs() < 1e-14);
        assert!((angles[1] - phi).abs() < 1e-14);
        // Unit conversion and JSON parsing can round the source f64 by an ulp.
        // Check the original radians numerically above, and require every engine
        // to preserve its ByteMessage round trip exactly below.
        let expected = Gate::rxyxy2q(
            message_angle(angles[0]),
            message_angle(angles[1]),
            &[(2, 0)],
        );
        let mut processor = OperationProcessor::new();
        processor.add_quantum_variable("q", 3).unwrap();
        let (name, qubits, angles) = processor
            .process_quantum_op(qop, Some(angles), args)
            .unwrap();
        let mut builder = pecos_engines::ByteMessage::quantum_operations_builder();
        processor
            .add_quantum_operation_to_builder(&mut builder, &name, &qubits, &angles)
            .unwrap();
        assert_eq!(
            builder.build().quantum_ops().unwrap(),
            std::slice::from_ref(&expected)
        );
        let mut engine = PhirJsonEngine::from_program(ast).unwrap();
        assert_eq!(
            engine.generate_commands().unwrap().quantum_ops().unwrap(),
            std::slice::from_ref(&expected)
        );
        let module = phir_json_to_module(&source).unwrap();
        let mut engine = PhirEngine::new(module).unwrap();
        assert_eq!(
            engine.generate_commands().unwrap().quantum_ops().unwrap(),
            std::slice::from_ref(&expected)
        );
    }
}

#[test]
fn rxyxy2q_json_rejects_wrong_counts() {
    for values in [json!([]), json!([-0.73]), json!([-0.73, 0.41, 0.2])] {
        let json = program(json!([values, "rad"]), json!([[["q", 2], ["q", 0]]]));
        assert!(serde_json::from_value::<PHIRProgram>(json.clone()).is_err());
        assert!(phir_json_to_module(&json.to_string()).is_err());
    }
    for args in [
        json!([]),
        json!([["q", 2]]),
        json!([[["q", 2], ["q", 0], ["q", 1]]]),
    ] {
        let json = program(json!([[-0.73, 0.41], "rad"]), args);
        assert!(serde_json::from_value::<PHIRProgram>(json.clone()).is_err());
        assert!(phir_json_to_module(&json.to_string()).is_err());
    }
    let processor = OperationProcessor::new();
    for angles in [vec![], vec![-0.73], vec![-0.73, 0.41, 0.2]] {
        let mut builder = pecos_engines::ByteMessage::quantum_operations_builder();
        assert!(
            processor
                .add_quantum_operation_to_builder(&mut builder, "RXYXY2Q", &[2, 0], &angles)
                .is_err()
        );
    }
}

#[test]
fn rxyxy2q_json_preserves_multiple_pairs() {
    let json = program(
        serde_json::json!([[-0.73, 0.41], "rad"]),
        serde_json::json!([[["q", 2], ["q", 0]], [["q", 0], ["q", 1]]]),
    );
    let source = json.to_string();
    let ast: PHIRProgram = serde_json::from_str(&source).unwrap();
    let Operation::QuantumOp {
        angles: Some(angles),
        ..
    } = &ast.ops[1]
    else {
        panic!("expected rotation")
    };
    let theta = message_angle(angles[0]);
    let phi = message_angle(angles[1]);
    let module = phir_json_to_module(&source).unwrap();
    let mut function = pecos_phir::phir::Function::new(
        "main",
        pecos_phir::types::FunctionType {
            inputs: vec![],
            outputs: vec![],
            variadic: false,
        },
    );
    function.body = vec![module.body.clone()];
    let use_def = pecos_phir::analysis::UseDefInfo::compute(&function);
    let mut definitions = std::collections::BTreeSet::new();
    for (inst_idx, instruction) in module.body.blocks[0].operations.iter().enumerate() {
        for result in &instruction.results {
            assert_eq!(
                use_def.get_definition(result),
                Some(&pecos_phir::analysis::InstructionRef {
                    region_idx: 0,
                    block_idx: 0,
                    inst_idx,
                }),
            );
            assert!(
                definitions.insert(*result),
                "duplicate SSA definition: {result:?}"
            );
        }
    }
    let rotations: Vec<_> = module.body.blocks[0]
        .operations
        .iter()
        .filter(|instruction| {
            matches!(
                instruction.operation,
                pecos_phir::ops::Operation::Quantum(pecos_phir::ops::QuantumOp::RXYXY2Q(..))
            )
        })
        .collect();
    assert_eq!(rotations.len(), 2);
    assert!(
        rotations
            .iter()
            .all(|instruction| !instruction.results.is_empty())
    );
    let mut engine = PhirEngine::new(module).unwrap();
    let expected: Vec<_> = [(2, 0), (0, 1)]
        .into_iter()
        .map(|pair| Gate::rxyxy2q(theta, phi, &[pair]))
        .collect();
    assert_eq!(
        engine.generate_commands().unwrap().quantum_ops().unwrap(),
        expected
    );
}

#[test]
fn rxyxy2q_json_rejects_mixed_or_flat_grouping() {
    for args in [
        json!([["q", 0], [["q", 1], ["q", 2]], ["q", 3]]),
        json!([["q", 2], ["q", 0]]),
    ] {
        let mut json = program(json!([[-0.73, 0.41], "rad"]), args.clone());
        json["ops"][0]["size"] = json!(4);
        assert!(serde_json::from_value::<PHIRProgram>(json.clone()).is_err());
        assert!(phir_json_to_module(&json.to_string()).is_err());
        let args: Vec<pecos_phir_json::v0_1::ast::QubitArg> = serde_json::from_value(args).unwrap();
        let mut processor = OperationProcessor::new();
        processor.add_quantum_variable("q", 4).unwrap();
        assert!(
            processor
                .process_quantum_op("RXYXY2Q", Some(&vec![-0.73, 0.41]), &args)
                .is_err()
        );
    }
}
