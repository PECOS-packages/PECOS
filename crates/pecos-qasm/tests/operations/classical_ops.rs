//! Comprehensive tests for classical operations in QASM
//! Consolidates tests for basic, complex, and supported classical operations

use pecos_qasm::{Operation, engine::QASMEngine, parser::QASMParser};
use std::str::FromStr;

#[test]
fn test_basic_classical_assignments() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[1];
        creg c[4];
        creg a[2];
        creg b[3];

        // Basic assignments
        c = 2;              // Direct integer assignment
        c = a;              // Register to register assignment
        c[0] = 1;           // Bit assignment
        c[1] = a[0];        // Bit to bit assignment
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse basic classical operations");
    assert!(!program.operations.is_empty());
}

#[test]
fn test_classical_arithmetic_operations() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[1];
        creg a[4];
        creg b[4];
        creg c[4];

        // Arithmetic operations
        c = a + b;          // Addition
        c = a - b;          // Subtraction
        c = a * b;          // Multiplication
        c = a / b;          // Division (integer)
        c = a ^ b;          // XOR
        c = a & b;          // AND
        c = a | b;          // OR
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse arithmetic operations");
    assert!(!program.operations.is_empty());
}

#[test]
fn test_classical_bitwise_operations() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[1];
        creg a[8];
        creg b[8];
        creg c[8];

        // Bitwise operations
        c = ~a;             // NOT
        c = a << 1;         // Left shift
        c = a >> 2;         // Right shift
        c[0] = a[0] ^ 1;    // XOR with constant
        c[1] = ~a[1];       // NOT individual bit
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse bitwise operations");
    assert!(!program.operations.is_empty());
}

#[test]
fn test_classical_conditional_operations() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[2];
        creg c[4];
        creg a[2];
        creg b[3];

        // Complex conditional operations
        if (b != 2) c[1] = b[1] & a[1] | a[0];
        if (a == 0) x q[0];
        if (c > 5) x q[1];  // Simplified to single operation
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse conditional operations");

    // Verify conditional operations are parsed
    let has_conditionals = program
        .operations
        .iter()
        .any(|op| matches!(op, Operation::If { .. }));
    assert!(has_conditionals, "Should have conditional operations");
}

#[test]
fn test_complex_classical_expressions() {
    let qasm = r#"
        OPENQASM 2.0;
        include "hqslib1.inc";

        qreg q[1];
        creg c[4];
        creg a[2];
        creg b[3];
        creg d[1];

        // Complex expressions
        c = 2;
        c = a;
        c[1] = b[1] & a[1] | a[0];
        b = a + b;
        b[1] = b[0] + ~b[2];
        c = a - b;
        d = a << 1;
        d = c >> 2;
        b = a * c / b;
        d[0] = a[0] ^ 1;
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse complex expressions");

    // Count classical operations
    let classical_count = program
        .operations
        .iter()
        .filter(|op| matches!(op, Operation::ClassicalAssignment { .. }))
        .count();

    assert!(
        classical_count >= 10,
        "Should have many classical operations"
    );
}

#[test]
fn test_classical_operations_with_execution() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[1];
        creg c[4];
        creg a[2];

        // Test with actual execution
        c = 5;              // c = 0101
        a = 3;              // a = 11
        c = c + a;          // c = 0101 + 0011 = 1000 (8)
        c[0] = 0;           // c = 1000

        measure q[0] -> a[0];
    "#;

    // Test both parsing and execution
    let _engine = QASMEngine::from_str(qasm).expect("Failed to create engine");
    // Simply verify the engine was created successfully with the classical operations
    // More comprehensive testing happens in other tests that actually run the simulation
}

#[test]
fn test_supported_vs_unsupported_operations() {
    // Document what's supported vs not supported

    // SUPPORTED:
    let supported_qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        creg c[4];
        creg a[4];

        // These should all parse successfully
        c = 5;              // Integer assignment
        c = a;              // Register assignment
        c = a + 5;          // Arithmetic with constants
        c = a & 15;         // Bitwise with integer (hex not supported)
        c[0] = 1;           // Bit assignment
        c = ~a;             // Unary operations
    "#;

    match QASMParser::parse_str(supported_qasm) {
        Ok(_) => {} // Test passes
        Err(e) => panic!("All supported operations should parse, but got error: {e:?}"),
    }

    // UNSUPPORTED (if any):
    // Add tests for operations that should fail if there are known unsupported cases
}

fn undeclared_source(statement: &str) -> String {
    format!("OPENQASM 2.0; include \"qelib1.inc\"; qreg q[1]; creg c[1]; {statement}")
}

fn assert_undeclared_error(error: impl std::fmt::Display) {
    let message = error.to_string();
    assert!(message.contains("missing"), "{message}");
    assert!(message.contains("creg"), "{message}");
}

fn assert_gate_error(error: impl std::fmt::Display, expected: &[&str]) {
    let message = error.to_string();
    for fragment in expected {
        assert!(
            message.contains(fragment),
            "Expected {fragment:?}: {message}"
        );
    }
    assert!(
        !message.contains("creg"),
        "Gate errors must not suggest creg: {message}"
    );
}

macro_rules! gate_error_test {
    ($name:ident, $statement:literal, [$($expected:literal),+]) => {
        #[test]
        fn $name() {
            let source = undeclared_source($statement);
            assert_gate_error(QASMParser::parse_str(&source).expect_err("Invalid gate expression"), &[$($expected),+]);
        }
    };
}

macro_rules! undeclared_test {
    ($name:ident, $statement:literal) => {
        #[test]
        fn $name() {
            let source = undeclared_source($statement);
            assert_undeclared_error(
                QASMParser::parse_str(&source).expect_err("Must reject undeclared classical name"),
            );
        }
    };
}

undeclared_test!(
    undeclared_condition,
    "if(missing == 0) x q[0]; measure q[0] -> c[0];"
);
undeclared_test!(undeclared_condition_bit, "if(missing[0] == 0) x q[0];");
undeclared_test!(undeclared_condition_right, "if(c == missing) x q[0];");
undeclared_test!(undeclared_assignment_value, "c = missing;");
undeclared_test!(undeclared_assignment_bit_value, "c[0] = missing[0];");
undeclared_test!(undeclared_assignment_target, "missing = 0;");
undeclared_test!(undeclared_assignment_bit_target, "missing[0] = 0;");
undeclared_test!(undeclared_binary_left, "c = missing + c;");
undeclared_test!(undeclared_binary_right, "c = c + missing;");
undeclared_test!(undeclared_unary, "c = ~missing;");
undeclared_test!(undeclared_negation, "c = -missing[0];");
undeclared_test!(undeclared_nested_expression, "c = (1 + (missing[0] | c));");
undeclared_test!(undeclared_function_argument, "c = external(1, missing);");
undeclared_test!(
    undeclared_nested_function_argument,
    "c = external(other(~missing[0]));"
);
undeclared_test!(undeclared_builtin_argument, "c = sin(missing);");
undeclared_test!(undeclared_void_argument, "external(1, missing[0]);");
undeclared_test!(
    undeclared_condition_function_argument,
    "if(external(missing) == 0) x q[0];"
);
undeclared_test!(undeclared_measurement_target, "measure q[0] -> missing[0];");
undeclared_test!(
    undeclared_register_measurement_target,
    "measure q -> missing;"
);
undeclared_test!(undeclared_false_branch_value, "if(c == 1) c = missing;");
undeclared_test!(undeclared_false_branch_target, "if(c == 1) missing = 0;");
undeclared_test!(
    undeclared_false_branch_measurement,
    "if(c == 1) measure q[0] -> missing[0];"
);
gate_error_test!(
    undeclared_gate_argument,
    "rx(missing) q[0];",
    [
        "Cannot evaluate variable 'missing'",
        "gate arguments must be constant expressions"
    ]
);
gate_error_test!(
    undeclared_gate_bit_argument,
    "rx(missing[0]) q[0];",
    ["Bit references are not allowed in gate parameter expressions"]
);
gate_error_test!(
    undeclared_gate_function_argument,
    "rx(sin(missing)) q[0];",
    [
        "Cannot evaluate variable 'missing'",
        "gate arguments must be constant expressions"
    ]
);
gate_error_test!(
    undeclared_gate_body_argument,
    "gate custom(theta) a { rx(theta + missing) a; }",
    [
        "Gate 'custom'",
        "unknown parameter 'missing'",
        "declared parameters: [theta]"
    ]
);
gate_error_test!(
    undeclared_gate_body_bit_argument,
    "gate custom(theta) a { rx(missing[0]) a; }",
    [
        "Gate 'custom'",
        "bit references are not allowed in gate bodies",
        "missing[0]"
    ]
);
undeclared_test!(
    undeclared_gate_parameter_outside_scope,
    "gate custom(missing) a { rx(missing) a; } c = missing;"
);
undeclared_test!(
    undeclared_opaque_parameter_outside_scope,
    "opaque custom(missing) a; c = missing;"
);

gate_error_test!(
    undeclared_gate_qubit_as_value,
    "gate custom(theta) missing { rx(missing) missing; }",
    [
        "Gate 'custom'",
        "unknown parameter 'missing'",
        "declared parameters: [theta]"
    ]
);
gate_error_test!(
    undeclared_gate_parameter_in_other_scope,
    "gate first(missing) a { rx(missing) a; } gate second(theta) a { rx(missing) a; }",
    [
        "Gate 'second'",
        "unknown parameter 'missing'",
        "declared parameters: [theta]"
    ]
);
gate_error_test!(
    undeclared_replaced_gate_definition,
    "gate custom(theta) a { rx(missing) a; } gate custom(theta) a { rx(theta) a; }",
    [
        "Gate 'custom'",
        "unknown parameter 'missing'",
        "declared parameters: [theta]"
    ]
);

#[test]
fn undeclared_engine_reproducer() {
    let source = "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[1]; creg c[1]; if(i == 0) x q[0]; measure q[0] -> c[0];";
    let error = QASMEngine::from_str(source)
        .expect_err("Engine creation must fail before any shot")
        .to_string();
    assert!(
        error.contains("register: i") && error.contains("creg"),
        "{error}"
    );
}

#[test]
fn undeclared_raw_parser() {
    let error = QASMParser::parse_str_raw("OPENQASM 2.0; creg c[1]; c = missing;")
        .expect_err("Raw parsing must validate names");
    assert_undeclared_error(error);
}

#[test]
fn undeclared_unexpanded_parser() {
    let config = pecos_qasm::parser::ParseConfig {
        expand_gates: false,
        validate_gates: false,
        ..Default::default()
    };
    assert_undeclared_error(
        QASMParser::parse_with_config(&undeclared_source("c = missing;"), &config)
            .expect_err("Name resolution is required even without gate validation"),
    );
}

#[test]
fn undeclared_expansion_entrypoint() {
    assert_undeclared_error(
        QASMParser::preprocess_and_expand(&undeclared_source("external(missing);"))
            .expect_err("Expansion must validate standalone function arguments"),
    );
}

#[test]
fn undeclared_phir_json_source() {
    assert_undeclared_error(
        pecos_qasm::qasm_to_phir_json(&undeclared_source("if(missing == 0) x q[0];"))
            .expect_err("PHIR conversion must validate names"),
    );
}

#[test]
fn undeclared_phir_json_ast() {
    let mut program = QASMParser::parse_str(&undeclared_source("c = c;")).unwrap();
    program.classical_registers.remove("c");
    let error = pecos_qasm::program_to_phir_json(&program)
        .expect_err("Direct AST conversion must validate names");
    assert!(
        error.contains("register: c") && error.contains("creg"),
        "{error}"
    );
}

#[test]
fn declared_registers_and_formal_parameters_still_execute() {
    use pecos_engines::ClassicalControlEngineBuilder;
    use pecos_programs::Qasm;

    let source = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        gate inner(theta) a { rx(theta) a; }
        gate outer(theta) a { inner(theta) a; }
        opaque unused(theta) a;
        qreg q[1];
        creg c[2];
        creg i[2];
        i = 1;
        RNGseed(i);
        c = i + i[0];
        if(c == 2) outer(pi) q[0];
        if(i[0] == 0) x q[0];
        measure q[0] -> c[0];
    "#;
    let results = pecos_qasm::qasm_engine()
        .program(Qasm::from_string(source))
        .to_sim()
        .workers(1)
        .run(3)
        .unwrap();
    for shot in results.shots {
        assert_eq!(shot.data["c"].as_u32(), Some(3));
        assert_eq!(shot.data["i"].as_u32(), Some(1));
    }
    pecos_qasm::qasm_to_phir_json(source).expect("Declared names also convert to PHIR-JSON");
}

#[test]
fn undeclared_ast_engine_compile() {
    use pecos_engines::ClassicalEngine;
    let mut program = QASMParser::parse_str(&undeclared_source("if(c == 1) c = c;")).unwrap();
    program.classical_registers.remove("c");
    let engine = QASMEngine::new(pecos_qasm::QASMProgram::new(program, String::new()));
    let error = engine
        .compile()
        .expect_err("Compilation must validate even a false branch")
        .to_string();
    assert!(
        error.contains("register: c") && error.contains("creg"),
        "{error}"
    );
}

#[test]
fn undeclared_ast_engine_commands() {
    use pecos_engines::ClassicalEngine;
    let mut program =
        QASMParser::parse_str(&undeclared_source("creg d[1]; x q[0]; if(c == 1) d = 0;")).unwrap();
    program.classical_registers.remove("d");
    let mut engine = QASMEngine::new(pecos_qasm::QASMProgram::new(program, String::new()));
    let error = engine
        .generate_commands()
        .err()
        .expect("No quantum command may run before validation")
        .to_string();
    assert!(
        error.contains("register: d") && error.contains("creg"),
        "{error}"
    );
}

fn assert_evaluator_rejects_undeclared(expression: &pecos_qasm::Expression) {
    use bitvec::prelude::{BitVec, Lsb0};
    use pecos_qasm::bitvec_expression::{BitVecExpressionContext, evaluate_expression_bitvec};

    struct EmptyContext;
    impl BitVecExpressionContext for EmptyContext {
        fn get_register(&self, _: &str) -> Option<&BitVec<u8, Lsb0>> {
            None
        }
        fn get_register_size(&self, _: &str) -> Option<usize> {
            None
        }
    }
    assert_undeclared_error(
        evaluate_expression_bitvec(expression, &EmptyContext)
            .expect_err("The evaluator must not synthesize a value for an undeclared name"),
    );
}

#[test]
fn undeclared_evaluator_register() {
    assert_evaluator_rejects_undeclared(&pecos_qasm::Expression::Variable("missing".to_string()));
}

#[test]
fn undeclared_evaluator_bit() {
    assert_evaluator_rejects_undeclared(&pecos_qasm::Expression::BitId("missing".to_string(), 0));
}

#[cfg(feature = "phir")]
#[test]
fn undeclared_phir_module_source() {
    assert_undeclared_error(
        pecos_qasm::qasm_to_phir_module(&undeclared_source("if(missing == 0) x q[0];"))
            .expect_err("PHIR module conversion must validate names"),
    );
}

#[cfg(feature = "phir")]
#[test]
fn undeclared_phir_module_ast() {
    let mut program = QASMParser::parse_str(&undeclared_source("c = c;")).unwrap();
    program.classical_registers.remove("c");
    let error = pecos_qasm::qasm_program_to_phir_module(&program)
        .expect_err("Direct PHIR module conversion must validate names")
        .to_string();
    assert!(
        error.contains("register: c") && error.contains("creg"),
        "{error}"
    );
}

macro_rules! undeclared_bit_test {
    ($name:ident, $statement:literal) => {
        #[test]
        fn $name() {
            let error = QASMParser::parse_str(&undeclared_source($statement))
                .expect_err("A bit outside the register declaration must fail validation")
                .to_string();
            assert!(error.contains("c[1]") && error.contains("creg"), "{error}");
        }
    };
}

undeclared_bit_test!(undeclared_bit_condition, "if(c[1] == 0) x q[0];");
undeclared_bit_test!(undeclared_bit_assignment_value, "c = c[1];");
undeclared_bit_test!(undeclared_bit_assignment_target, "c[1] = 0;");
undeclared_bit_test!(undeclared_bit_binary_operand, "c = c + c[1];");
undeclared_bit_test!(undeclared_bit_unary_operand, "c = ~c[1];");
undeclared_bit_test!(undeclared_bit_function_argument, "c = external(c[1]);");
undeclared_bit_test!(undeclared_bit_void_argument, "external(c[1]);");
undeclared_bit_test!(undeclared_bit_measurement_target, "measure q[0] -> c[1];");
undeclared_bit_test!(undeclared_bit_false_branch, "if(c == 1) c[1] = 0;");
gate_error_test!(
    undeclared_bit_gate_argument,
    "rx(c[1]) q[0];",
    ["Bit references are not allowed in gate parameter expressions"]
);

#[test]
fn undeclared_evaluator_bit_out_of_bounds() {
    use bitvec::prelude::{BitVec, Lsb0};
    use pecos_qasm::bitvec_expression::{BitVecExpressionContext, evaluate_expression_bitvec};

    struct DeclaredContext;
    impl BitVecExpressionContext for DeclaredContext {
        fn get_register(&self, _: &str) -> Option<&BitVec<u8, Lsb0>> {
            None
        }
        fn get_register_size(&self, name: &str) -> Option<usize> {
            (name == "c").then_some(1)
        }
    }
    let error = evaluate_expression_bitvec(
        &pecos_qasm::Expression::BitId("c".to_string(), 1),
        &DeclaredContext,
    )
    .expect_err("Only declared bits may be read")
    .to_string();
    assert!(error.contains("c[1]") && error.contains("creg"), "{error}");
    assert!(
        !evaluate_expression_bitvec(
            &pecos_qasm::Expression::BitId("c".to_string(), 0),
            &DeclaredContext,
        )
        .unwrap()
        .as_bool()
    );
}

macro_rules! gate_scope_test {
    ($name:ident, $before:literal, $after:literal, $usage:literal) => {
        #[test]
        fn $name() {
            let source = undeclared_source(concat!(
                $before,
                " gate g(a) x0 { rz(b) x0; } ",
                $after,
                " ",
                $usage
            ));
            assert_gate_error(
                QASMParser::parse_str(&source).expect_err("Only gate parameters are in scope"),
                &[
                    "Gate 'g'",
                    "unknown parameter 'b'",
                    "declared parameters: [a]",
                ],
            );
        }
    };
}

gate_scope_test!(gate_scope_creg_before_used, "creg b[1];", "", "g(1) q[0];");
gate_scope_test!(gate_scope_creg_after_used, "", "creg b[1];", "g(1) q[0];");
gate_scope_test!(gate_scope_creg_absent_used, "", "", "g(1) q[0];");
gate_scope_test!(gate_scope_creg_before_unused, "creg b[1];", "", "");
gate_scope_test!(gate_scope_creg_after_unused, "", "creg b[1];", "");
gate_scope_test!(gate_scope_creg_absent_unused, "", "", "");

#[test]
fn gate_scope_direct_ast() {
    let mut program =
        QASMParser::parse_str(&undeclared_source("creg b[1]; gate g(a) x0 { rz(a) x0; }")).unwrap();
    program.gate_definitions.get_mut("g").unwrap().body[0].params[0] =
        pecos_qasm::Expression::Variable("b".to_string());
    assert_gate_error(
        program
            .validate_classical_references()
            .expect_err("AST validation must also use gate scope"),
        &[
            "Gate 'g'",
            "unknown parameter 'b'",
            "declared parameters: [a]",
        ],
    );
}

gate_error_test!(
    gate_scope_nested_argument,
    "creg b[1]; gate g(a) x0 { rz(sin(-b + a)) x0; }",
    [
        "Gate 'g'",
        "unknown parameter 'b'",
        "declared parameters: [a]"
    ]
);
gate_error_test!(
    gate_scope_declared_bit_argument,
    "creg b[1]; gate g(b) x0 { rz(b[0]) x0; }",
    [
        "Gate 'g'",
        "bit references are not allowed in gate bodies",
        "b[0]"
    ]
);
gate_error_test!(
    gate_scope_no_parameters,
    "gate g x0 { rz(b) x0; }",
    [
        "Gate 'g'",
        "unknown parameter 'b'",
        "declared parameters: []"
    ]
);
