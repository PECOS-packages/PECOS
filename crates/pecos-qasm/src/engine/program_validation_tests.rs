//! Parser and engine checks: keyword boundaries, remainder, assignment selectors,
//! measurement targets, declaration order and `JOB_shotnum`.
use super::*;
use crate::parser::QASMParser;
use crate::qasm_to_phir_json::{program_to_phir_json, qasm_to_phir_json};
use pecos_core::prelude::{Gate, QubitId};
use serde_json::json;

fn run(source: &str) -> Result<QASMEngine, PecosError> {
    let mut engine = QASMEngine::new(QASMProgram::from_str(source)?);
    engine.allow_complex_conditionals(true);
    engine.process_program_impl()?;
    Ok(engine)
}

#[test]
fn keyword_boundaries() {
    for name in [
        "pivot",
        "pi_2",
        "pi2",
        "OPENQASM2",
        "include2",
        "qreg2",
        "creg2",
        "measure2",
        "reset2",
        "barrier2",
        "if2",
        "gate2",
        "opaque2",
    ] {
        let engine = run(&format!(
            "creg {name}[8]; {name} = 2; if({name} == 2) {name} = {name} + 1;"
        ))
        .unwrap();
        assert_eq!(
            pecos_core::bitvec::to_u64(&engine.classical_registers[name]),
            Some(3)
        );
        let program = QASMParser::parse_str(&format!(
            "qreg {name}[1]; gate g({name}) a {{ RZ({name}) a; }} g(pi) {name}[0];"
        ))
        .unwrap();
        let Operation::NativeGate(gate) = &program.operations[0] else {
            panic!("Expected expanded RZ gate");
        };
        assert!((gate.angles[0].to_radians() - std::f64::consts::PI).abs() < 1e-12);
    }
}

#[test]
fn remainder_runtime_and_folding() {
    let engine = run(
        "creg a[8]; creg b[8]; creg c[8]; creg x[8]; creg y[8]; creg z[8];
         a = 5; b = 8; c = 3; x = a + b % c; y = (a + b) % c;
         z = 20 % 6 * 3;",
    )
    .unwrap();
    for (name, expected) in [("x", 7), ("y", 1), ("z", 6)] {
        assert_eq!(
            pecos_core::bitvec::to_u64(&engine.classical_registers[name]),
            Some(expected)
        );
    }
    for expression in ["7 % 0", "a % 0"] {
        assert!(matches!(
            run(&format!("creg a[8]; a = 7; a = {expression};")),
            Err(PecosError::RuntimeDivisionByZero)
        ));
    }
    let engine = run("creg s[100]; creg w[100]; s = -7 % 3; w[99] = 1; w = w % 3;").unwrap();
    assert!(engine.classical_registers["s"].all());
    assert_eq!(
        pecos_core::bitvec::to_u64(&engine.classical_registers["w"]),
        Some(2)
    );
    let output = qasm_to_phir_json("creg a[8]; creg b[8]; a = a % b;").unwrap();
    assert!(
        output["ops"]
            .as_array()
            .unwrap()
            .iter()
            .any(|op| { op["args"] == json!([{ "cop": "%", "args": ["a", "b"] }]) })
    );
}

#[test]
fn assignment_index_is_the_only_state() {
    let mut program = QASMParser::parse_str("creg c[8];").unwrap();
    // Intentionally exhaustive construction: restoring the redundant field must not compile.
    let whole = Operation::ClassicalAssignment {
        target: "c".into(),
        index: None,
        expression: Expression::Integer(
            crate::parser::expressions::parse_integer_to_bitvec("4").unwrap(),
        ),
    };
    let indexed = Operation::ClassicalAssignment {
        target: "c".into(),
        index: Some(7),
        expression: Expression::Integer(
            crate::parser::expressions::parse_integer_to_bitvec("1").unwrap(),
        ),
    };
    assert_eq!(whole.to_string(), "c = 4");
    assert_eq!(indexed.to_string(), "c[7] = 1");
    program.operations = vec![
        whole,
        Operation::If {
            condition: Expression::Integer(
                crate::parser::expressions::parse_integer_to_bitvec("1").unwrap(),
            ),
            operation: Box::new(indexed),
        },
    ];
    let json = program_to_phir_json(&program).unwrap();
    let ops = json["ops"].as_array().unwrap();
    assert!(ops.iter().any(|op| op["returns"] == json!(["c"])));
    assert!(
        ops.iter()
            .any(|op| op["true_branch"][0]["returns"] == json!([["c", 7]]))
    );
    let mut engine = QASMEngine::new(QASMProgram::new(program, String::new()));
    engine.allow_complex_conditionals(true);
    engine.process_program_impl().unwrap();
    assert_eq!(
        pecos_core::bitvec::to_u64(&engine.classical_registers["c"]),
        Some(132)
    );
    let engine = run("creg c[8]; c[7] = 1; if(1) c = 4;").unwrap();
    assert_eq!(
        pecos_core::bitvec::to_u64(&engine.classical_registers["c"]),
        Some(4)
    );
}

#[test]
fn empty_measurement_name_is_unknown() {
    let program = QASMProgram::from_str("qreg q[1]; creg c[1];").unwrap();
    let mut engine = QASMEngine::new(program.clone());
    for name in ["", "missing"] {
        let message = engine
            .process_measurement(0, name, 0)
            .unwrap_err()
            .to_string();
        assert!(
            message.contains(&format!("'{name}' not found")),
            "{message}"
        );
        assert!(
            engine
                .process_register_measurement("q", name, &program, None)
                .is_err()
        );
        for operation in [
            Operation::MeasureWithMapping {
                gate: Gate::mz(&[QubitId::from(0)]),
                c_reg: name.into(),
                c_index: 0,
            },
            Operation::RegMeasure {
                q_reg: "q".into(),
                c_reg: name.into(),
            },
        ] {
            let mut ast = program.program().clone();
            ast.operations.push(operation);
            assert!(ast.validate_classical_references().is_err());
            assert!(program_to_phir_json(&ast).is_err());
        }
    }
    assert_eq!(
        engine.register_result_mappings,
        Vec::<(String, usize)>::new()
    );
}

#[test]
fn classical_declarations_precede_every_use() {
    for statement in [
        "if(later == 0) X q[0];",
        "if(later[0] == 0) X q[0];",
        "later = 0;",
        "later[0] = 0;",
        "c = later;",
        "c = later[0];",
        "c = later + c;",
        "c = c + later;",
        "c = ~later;",
        "c = -later[0];",
        "c = external(1, later);",
        "c = external(nested(later[0]));",
        "external(later);",
        "if(external(later)) c = 0;",
        "measure q[0] -> later[0];",
        "measure q -> later;",
        "if(c) later = 0;",
        "if(c) c = later;",
        "if(c) measure q -> later;",
    ] {
        let prefix = "qreg q[1]; creg c[1];";
        let error = QASMParser::parse_str(&format!("{prefix} {statement} creg later[1];"))
            .unwrap_err()
            .to_string();
        for fragment in ["later", "creg", "before its first use"] {
            assert!(error.contains(fragment), "{statement}: {error}");
        }
        QASMParser::parse_str(&format!("{prefix} creg later[1]; {statement}")).unwrap();
    }
}

#[test]
fn shot_number_has_a_specific_error() {
    for source in [
        include_str!(
            "../../../../python/quantum-pecos/tests/pecos/regression/test_qasm/regression_qasm/Job_shotnum_usage.qasm"
        ),
        include_str!(
            "../../../../python/quantum-pecos/tests/pecos/regression/test_qasm/regression_qasm/RNGnum_jobvar_h5.qasm"
        ),
    ] {
        assert_shot_number_error(&QASMParser::parse_str(source).unwrap_err());
    }
    for statement in [
        "c = JOB_shotnum;",
        "c = JOB_shotnum[0];",
        "JOB_shotnum = 0;",
        "JOB_shotnum[0] = 0;",
        "if(JOB_shotnum) X q[0];",
        "external(JOB_shotnum);",
        "c = external(JOB_shotnum);",
        "c = external(nested(JOB_shotnum[0]));",
        "c = c + JOB_shotnum;",
        "c = JOB_shotnum + c;",
        "c = ~JOB_shotnum;",
        "if(c) c = JOB_shotnum;",
        "if(c) JOB_shotnum[0] = 0;",
        "if(c) measure q[0] -> JOB_shotnum[0];",
        "measure q -> JOB_shotnum;",
        "measure q[0] -> JOB_shotnum[0];",
    ] {
        assert_shot_number_error(
            &QASMParser::parse_str(&format!("qreg q[1]; creg c[1]; {statement}")).unwrap_err(),
        );
    }
    let engine = run("creg JOB_shotnum[8]; creg c[8]; c = JOB_shotnum; JOB_shotnum = 7;").unwrap();
    assert_eq!(
        pecos_core::bitvec::to_u64(&engine.classical_registers["c"]),
        Some(0)
    );
    assert_eq!(
        pecos_core::bitvec::to_u64(&engine.classical_registers["JOB_shotnum"]),
        Some(7)
    );
    assert_shot_number_error(
        &crate::bitvec_expression::evaluate_expression_bitvec(
            &Expression::Variable("JOB_shotnum".into()),
            &run("creg c[1];").unwrap(),
        )
        .unwrap_err(),
    );
}

fn assert_shot_number_error(error: &PecosError) {
    let message = error.to_string();
    for fragment in [
        "JOB_shotnum",
        "not supported",
        "PECOS Rust QASM engine",
        "shot-number built-in",
        "Remove the reference to JOB_shotnum",
    ] {
        assert!(message.contains(fragment), "{message}");
    }
    assert!(!message.contains("creg"), "{message}");
}

#[test]
fn remainder_gate_arguments() {
    use crate::parser::constant_folding::fold_constants_gate_param;
    for (op, value) in [("/", 3.5), ("%", 1.0)] {
        let expression = Expression::BinaryOp {
            op: op.into(),
            left: Box::new(Expression::Float(7.0)),
            right: Box::new(Expression::Float(2.0)),
        };
        assert!((expression.evaluate(None).unwrap() - value).abs() < 1e-12);
        let Expression::Float(folded) = fold_constants_gate_param(expression) else {
            panic!("Gate arithmetic should fold");
        };
        assert!((folded - value).abs() < 1e-12);
        let zero_divisor = Expression::BinaryOp {
            op: op.into(),
            left: Box::new(Expression::Float(7.0)),
            right: Box::new(Expression::Float(0.0)),
        };
        // Like division, zero remains unfolded and evaluates with IEEE semantics.
        let folded = fold_constants_gate_param(zero_divisor);
        assert!(matches!(folded, Expression::BinaryOp { .. }));
        let result = folded.evaluate(None).unwrap();
        if op == "/" {
            assert!(result.is_infinite());
        } else {
            assert!(result.is_nan());
        }
    }
    let program =
        QASMParser::parse_str("qreg q[1]; gate g(a) b { RZ(a % 2) b; } g(7) q[0]; RZ(7 % 2) q[0];")
            .unwrap();
    for operation in program.operations {
        let Operation::NativeGate(gate) = operation else {
            panic!("Expected native RZ");
        };
        assert!((gate.angles[0].to_radians() - 1.0).abs() < 1e-12);
    }
}

#[test]
fn glued_statement_keywords_are_syntax_errors() {
    for statement in [
        "qregq[1];",
        "cregc[1];",
        "measureq[0]->c[0];",
        "resetq[0];",
        "barrierq;",
    ] {
        let source = format!("qreg q[1]; creg c[1]; {statement}");
        assert!(
            matches!(
                QASMParser::parse_str(&source),
                Err(PecosError::ParseSyntax { .. })
            ),
            "Expected a syntax error for {statement}"
        );
    }
}

#[test]
fn version_header_accepts_no_space() {
    for source in ["OPENQASM2.0;", "OPENQASM 2.0;", "OPENQASM2.0; qreg q[1];"] {
        assert_eq!(QASMParser::parse_str(source).unwrap().version, "2.0");
    }
}

#[test]
fn classical_bit_indices_are_bounds_checked() {
    for statement in [
        "c[7] = 1;",
        "out = c[1];",
        "out = 1 + c[1];",
        "out = ~c[1];",
        "measure q[1] -> c[1];",
        "if(c[1] == 1) X q[0];",
        "external(c[1]);",
        "out = external(c[1]);",
        "if(out) c[1] = 1;",
        "if(out) measure q[1] -> c[1];",
    ] {
        let source = format!("qreg q[2]; creg out[8]; creg c[1]; {statement}");
        let error = QASMParser::parse_str(&source).unwrap_err().to_string();
        assert!(error.contains("Classical bit 'c["), "{statement}: {error}");
        assert!(
            error.contains("out of bounds for register size 1"),
            "{statement}: {error}"
        );
        QASMParser::parse_str(&format!("qreg q[2]; creg out[8]; creg c[8]; {statement}")).unwrap();
    }
}

#[test]
fn shot_number_is_an_ordinary_quantum_or_gate_name() {
    for body in [
        "qreg JOB_shotnum[1]; h JOB_shotnum[0];",
        "gate g(JOB_shotnum) a { rz(JOB_shotnum) a; } qreg q[1]; g(pi) q[0];",
        "gate JOB_shotnum a { h a; } qreg q[1]; JOB_shotnum q[0];",
    ] {
        QASMParser::parse_str(&format!("include \"qelib1.inc\"; {body}")).unwrap();
    }
}

#[test]
fn shot_number_in_gate_expressions_gets_ordinary_errors() {
    // Gate parameters named JOB_shotnum resolve like any other name.
    for statement in [
        "RZ(JOB_shotnum) q[0];",
        "RZ(sin(JOB_shotnum)) q[0];",
        "gate g(a) b { RZ(JOB_shotnum) b; }",
    ] {
        let source = format!("qreg q[1]; {statement}");
        let error = QASMParser::parse_str(&source).unwrap_err().to_string();
        let ordinary = QASMParser::parse_str(&source.replace("JOB_shotnum", "ordinary"))
            .unwrap_err()
            .to_string();
        assert_eq!(error.replace("JOB_shotnum", "ordinary"), ordinary);
        assert!(!error.contains("shot-number built-in"), "{error}");
    }
}

#[test]
fn caller_built_programs_are_bounds_checked() {
    // Caller-built ASTs still need explicit reference and bounds validation.
    let mut program = QASMParser::parse_str("creg d[8]; d[7] = 1;").unwrap();
    program.classical_registers.insert("d".into(), 1);
    assert_eq!(
        program
            .validate_classical_references()
            .unwrap_err()
            .to_string(),
        "Invalid QASM operation: Classical bit 'd[7]' is out of bounds for register size 1. Declare 'd' with creg large enough to contain bit 7"
    );
}

#[test]
fn glued_gate_is_not_a_definition() {
    assert!(matches!(
        QASMParser::parse_str("gateg a { }"),
        Err(PecosError::ParseSyntax { .. })
    ));
}

#[test]
fn glued_opaque_is_a_gate_call() {
    use crate::parser::Rule;
    use pest::Parser;
    assert_eq!(
        QASMParser::parse_str("opaqueo a;").unwrap_err().to_string(),
        "Invalid QASM operation: Unknown quantum register: a"
    );
    // The full grammar tries gate calls first; check the declaration rule as well.
    assert!(QASMParser::parse(Rule::opaque_def, "opaqueo a;").is_err());
    assert!(QASMParser::parse(Rule::opaque_def, "opaque o a;").is_ok());
}

#[test]
fn glued_if_is_not_a_conditional() {
    // `if_stmt` requires `(` right after `if`, so this documents the behaviour
    // rather than guarding the keyword lookahead.
    use crate::parser::Rule;
    use pest::Parser;
    assert!(matches!(
        QASMParser::parse_str("qreg q[1]; ifc(1) X q[0];"),
        Err(PecosError::ParseSyntax { .. })
    ));
    assert!(QASMParser::parse(Rule::if_stmt, "ifc(1) X q[0];").is_err());
    assert!(QASMParser::parse(Rule::if_stmt, "if(1) X q[0];").is_ok());
}

#[test]
fn register_redeclarations_are_rejected() {
    for (first, second, kind) in [
        ("creg c[1];", "creg c[8]; c[7] = 1;", "creg"),
        ("creg c[8]; c[7] = 1;", "creg c[1];", "creg"),
        ("creg c[1];", "creg c[1];", "creg"),
        ("qreg c[1];", "qreg c[8]; X c[7];", "qreg"),
        ("qreg c[8]; X c[7];", "qreg c[1];", "qreg"),
        ("qreg c[1];", "qreg c[1];", "qreg"),
        ("creg c[1];", "qreg c[1];", "creg"),
        ("qreg c[1];", "creg c[1];", "qreg"),
    ] {
        let source = format!("{first} {second}");
        let error = QASMParser::parse_str(&source).unwrap_err().to_string();
        let expected = format!(
            "Register 'c' is already declared as {kind}. Declare each register once, or use a different name"
        );
        assert!(error.contains(&expected), "{source}: {error}");
    }
}

#[test]
fn rejected_redeclaration_does_not_mutate_program() {
    use crate::parser::{Program, Rule, registers::parse_register};
    use pest::Parser;
    for first_kind in ["qreg", "creg"] {
        for second_kind in ["qreg", "creg"] {
            let mut program = Program::default();
            let first = format!("{first_kind} r[8];");
            let pair = QASMParser::parse(Rule::register_decl, &first)
                .unwrap()
                .next()
                .unwrap();
            parse_register(pair, &mut program).unwrap();
            let before = format!("{program:?}");
            let second = format!("{second_kind} r[1];");
            let pair = QASMParser::parse(Rule::register_decl, &second)
                .unwrap()
                .next()
                .unwrap();
            assert_eq!(
                parse_register(pair, &mut program).unwrap_err().to_string(),
                format!(
                    "Invalid QASM operation: Register 'r' is already declared as {first_kind}. Declare each register once, or use a different name"
                )
            );
            assert_eq!(format!("{program:?}"), before);
        }
    }
}
