#![cfg(feature = "phir")]

use pecos_core::{Angle64, Gate};
use pecos_engines::ClassicalEngine;
use pecos_phir::PhirEngine;
use pecos_phir_json::phir_json_to_module;
use pecos_phir_json::v0_1::ast::{Operation, PHIRProgram};
use pecos_qasm::{QASMEngine, qasm_to_phir_json, qasm_to_phir_module};
use std::str::FromStr;

#[test]
fn rxyxy2q_qasm_phir_equivalence() {
    for gate in ["RXYXY2Q", "rxyxy2q"] {
        let qasm = format!(
            "OPENQASM 2.0; include \"pecos.inc\"; qreg q[3]; {gate}(-0.73, 0.41) q[2], q[0];"
        );
        let mut direct = QASMEngine::from_str(&qasm).unwrap();
        let expected = direct.generate_commands().unwrap().quantum_ops().unwrap();
        assert_eq!(
            expected,
            [Gate::rxyxy2q(
                Angle64::from_radians(-0.73),
                Angle64::from_radians(0.41),
                &[(2, 0)]
            )]
        );
        let json = qasm_to_phir_json(&qasm).unwrap();
        let ast: PHIRProgram = serde_json::from_value(json.clone()).unwrap();
        let Operation::QuantumOp {
            qop,
            angles: Some(angles),
            args,
            ..
        } = &ast.ops[1]
        else {
            panic!("expected rotation")
        };
        let module = phir_json_to_module(&json.to_string()).unwrap();
        let mut engine = PhirEngine::new(module).unwrap();
        assert_eq!(
            engine.generate_commands().unwrap().quantum_ops().unwrap(),
            expected
        );
        assert_eq!(qop, "RXYXY2Q");
        assert_eq!(Angle64::from_radians(angles[0]), expected[0].angles[0]);
        assert_eq!(Angle64::from_radians(angles[1]), expected[0].angles[1]);
        let [pecos_phir_json::v0_1::ast::QubitArg::MultipleQubits(pair)] = args.as_slice() else {
            panic!("expected pair")
        };
        assert_eq!(pair, &vec![("q".to_string(), 2), ("q".to_string(), 0)]);
        let module = qasm_to_phir_module(&qasm).unwrap();
        let mut engine = PhirEngine::new(module).unwrap();
        assert_eq!(
            engine.generate_commands().unwrap().quantum_ops().unwrap(),
            expected
        );
    }
}

#[test]
fn rxyxy2q_bridges_reject_wrong_counts() {
    use pecos_qasm::ast::Operation as QasmOp;
    use pecos_qasm::parser::QASMParser;
    let qasm = "OPENQASM 2.0; qreg q[3]; RXYXY2Q(-0.73, 0.41) q[2], q[0];";
    let original = QASMParser::parse_str(qasm).unwrap();
    for count in [0, 1, 3] {
        let mut program = original.clone();
        let QasmOp::NativeGate(gate) = &mut program.operations[0] else {
            panic!("expected native gate")
        };
        gate.angles = vec![Angle64::from_radians(-0.73); count].into();
        assert!(pecos_qasm::program_to_phir_json(&program).is_err());
        assert!(pecos_qasm::qasm_program_to_phir_module(&program).is_err());
    }
    for qubits in [vec![], vec![2.into()], vec![2.into(), 0.into(), 1.into()]] {
        let mut program = original.clone();
        let QasmOp::NativeGate(gate) = &mut program.operations[0] else {
            panic!("expected native gate")
        };
        gate.qubits = qubits.into();
        assert!(pecos_qasm::program_to_phir_json(&program).is_err());
        assert!(pecos_qasm::qasm_program_to_phir_module(&program).is_err());
    }
}
