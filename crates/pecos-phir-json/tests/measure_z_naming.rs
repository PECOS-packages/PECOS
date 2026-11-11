use pecos_engines::{ClassicalEngine, Engine, GateType, StateVecEngine};
use pecos_phir::PhirEngine;
use pecos_phir_json::PhirJsonEngine;
use pecos_phir_json::PhirJsonEngineProgram;
use pecos_phir_json::phir_json_to_module;
use pecos_phir_json::v0_1::ast::{Operation, PHIRProgram};
use pecos_phir_json::v0_1::classical_interpreter::PhirClassicalInterpreter;
use serde_json::{Value, json};

fn program(name: &str) -> Value {
    json!({"format": "PHIR/JSON", "version": "0.1.0", "ops": [
        {"data": "qvar_define", "variable": "q", "size": 1},
        {"data": "cvar_define", "variable": "m", "data_type": "u32", "size": 1},
        {"qop": "X", "args": [["q", 0]]},
        {"qop": name, "args": [["q", 0]], "returns": [["m", 0]]}
    ]})
}

#[test]
fn measure_z_is_canonicalized_before_execution() {
    let mut command_bytes = Vec::new();
    let mut results = Vec::new();
    for name in ["Measure", "MeasureZ"] {
        let source = program(name).to_string();
        let ast: PHIRProgram = serde_json::from_str(&source).unwrap();
        let Operation::QuantumOp { qop, .. } = &ast.ops[3] else {
            panic!("expected measurement");
        };
        assert_eq!(qop, "Measure");
        let mut engine = PhirJsonEngine::from_json(&source).unwrap();
        let commands = engine.generate_commands().unwrap();
        assert_eq!(commands.quantum_ops().unwrap()[1].gate_type, GateType::MZ);
        command_bytes.push(commands.as_bytes().to_vec());
        let outcomes = StateVecEngine::new(1).process(commands).unwrap();
        assert_eq!(outcomes.outcomes().unwrap(), [1]);
        engine.handle_measurements(outcomes).unwrap();
        assert!(engine.generate_commands().unwrap().is_empty().unwrap());
        results.push(engine.get_results().unwrap().data);

        let mut converted = PhirEngine::new(phir_json_to_module(&source).unwrap()).unwrap();
        let commands = converted.generate_commands().unwrap();
        assert_eq!(commands.quantum_ops().unwrap()[1].gate_type, GateType::MZ);
        let outcomes = StateVecEngine::new(1).process(commands).unwrap();
        assert_eq!(outcomes.outcomes().unwrap(), [1]);
        converted.handle_measurements(outcomes).unwrap();
    }
    assert_eq!(command_bytes[0], command_bytes[1]);
    assert_eq!(results[0], results[1]);
}

#[test]
fn nested_measure_z_is_canonicalized() {
    for field in ["ops", "true_branch", "false_branch"] {
        let mut source = program("MeasureZ");
        let measure = source["ops"].as_array_mut().unwrap().pop().unwrap();
        source["ops"].as_array_mut().unwrap().push(json!({
            "block": if field == "ops" { "sequence" } else { "if" },
            "condition": {"cop": "==", "args": [0, 0]}, field: [measure]
        }));
        let source = source.to_string();
        // The module converter currently rejects blocks for either spelling.
        assert_eq!(
            phir_json_to_module(&source).unwrap_err().to_string(),
            phir_json_to_module(&source.replace("MeasureZ", "Measure"))
                .unwrap_err()
                .to_string()
        );
        let ast: PHIRProgram = serde_json::from_str(&source).unwrap();
        let Operation::Block {
            ops,
            true_branch,
            false_branch,
            ..
        } = &ast.ops[3]
        else {
            panic!("expected block");
        };
        let nested = match field {
            "true_branch" => true_branch.as_ref().unwrap(),
            "false_branch" => false_branch.as_ref().unwrap(),
            _ => ops,
        };
        assert!(matches!(&nested[0], Operation::QuantumOp { qop, .. } if qop == "Measure"));
    }
}

#[test]
fn measure_zed_is_an_unknown_gate() {
    for name in ["MeasureZed", "UnknownGate"] {
        let mut engine = PhirJsonEngine::from_json(&program(name).to_string()).unwrap();
        assert!(engine.generate_commands().is_err());
        assert!(phir_json_to_module(&program(name).to_string()).is_err());
    }
}

fn assert_malformed_measurement_errors_match(
    parse: impl Fn(&str) -> Result<(), pecos_core::errors::PecosError>,
) {
    for field in [None, Some("ops"), Some("true_branch"), Some("false_branch")] {
        let error = |name| {
            let mut source = program(name);
            source["ops"][0]["size"] = json!(2);
            source["ops"][1]["size"] = json!(2);
            source["ops"][3]["args"] = json!([["q", 0], ["q", 1]]);
            if let Some(field) = field {
                let measure = source["ops"].as_array_mut().unwrap().pop().unwrap();
                source["ops"].as_array_mut().unwrap().push(json!({
                    "block": if field == "ops" { "sequence" } else { "if" },
                    "condition": {"cop": "==", "args": [0, 0]}, field: [measure]
                }));
            }
            parse(&source.to_string()).expect_err("measurement return count must be rejected")
        };
        let expected = error("Measure");
        let actual = error("MeasureZ");
        assert_eq!(
            std::mem::discriminant(&actual),
            std::mem::discriminant(&expected)
        );
        assert_eq!(actual.to_string(), expected.to_string());
    }
}

#[test]
fn malformed_measure_z_program_diagnostics() {
    assert_malformed_measurement_errors_match(|source| PHIRProgram::from_json(source).map(|_| ()));
}

#[test]
fn malformed_measure_z_engine_diagnostics() {
    assert_malformed_measurement_errors_match(|source| {
        PhirJsonEngine::from_json(source).map(|_| ())
    });
}

#[test]
fn malformed_measure_z_builder_diagnostics() {
    assert_malformed_measurement_errors_match(|source| {
        PhirJsonEngineProgram::from_json(source).map(|_| ())
    });
}

#[test]
fn malformed_measure_z_classical_interpreter_diagnostics() {
    assert_malformed_measurement_errors_match(|source| {
        PhirClassicalInterpreter::new()
            .init(source, None)
            .map(|_| ())
    });
}
