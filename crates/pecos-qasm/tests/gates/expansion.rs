use pecos_core::errors::PecosError;
use pecos_core::prelude::{Gate, GateType, QubitId};
use pecos_engines::sim_builder;
use pecos_programs::Qasm;
use pecos_qasm::parser::{ParseConfig, QASMParser};
use pecos_qasm::{Operation, qasm_engine};

// Helper function to check if an operation is a specific gate
fn is_gate_with_name(op: &Operation, gate_name: &str) -> bool {
    match op {
        Operation::Gate { name, .. } => name == gate_name,
        Operation::NativeGate(gate) => match gate_name {
            "H" => matches!(gate.gate_type, GateType::H),
            "X" => matches!(gate.gate_type, GateType::X),
            "CX" => matches!(gate.gate_type, GateType::CX),
            _ => false,
        },
        _ => false,
    }
}

#[test]
fn test_gate_expansion_basic() {
    let qasm = r"
        OPENQASM 2.0;
        qreg q[1];

        gate mygate a { H a; }

        mygate q[0];
    ";

    let program = QASMParser::parse_str_raw(qasm).unwrap();

    // Gate definition should be loaded
    assert!(program.gate_definitions.contains_key("mygate"));

    // The mygate operation should be expanded to H
    assert_eq!(program.operations.len(), 1);

    assert!(
        is_gate_with_name(&program.operations[0], "H"),
        "Expected H gate"
    );
}

#[test]
fn test_gate_expansion_native_gate() {
    let qasm = r"
        OPENQASM 2.0;
        qreg q[1];
        H q[0];
    ";

    let program = QASMParser::parse_str_raw(qasm).unwrap();

    // Native gate should not be expanded
    assert_eq!(program.operations.len(), 1);

    assert!(
        is_gate_with_name(&program.operations[0], "H"),
        "Expected H gate"
    );
}

#[test]
fn test_gate_expansion_rx() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        rx(pi/2) q[0];
    "#;

    let program = QASMParser::parse_str(qasm).unwrap();

    // The rx gate should lower directly to RXY1Q(pi/2, 0).
    assert_eq!(program.operations.len(), 1);

    match &program.operations[0] {
        Operation::NativeGate(gate) => {
            assert_eq!(gate.gate_type, GateType::RXY1Q);
            assert_eq!(gate.qubits.len(), 1);
            assert_eq!({ gate.qubits[0].0 }, 0);
            assert_eq!(gate.angles.len(), 2);
            assert!(
                (gate.angles[0].to_radians() - std::f64::consts::FRAC_PI_2).abs() < 1e-6,
                "Expected theta PI/2, got {}",
                gate.angles[0].to_radians()
            );
            assert!(
                gate.angles[1].to_radians().abs() < 1e-6,
                "Expected phi 0, got {}",
                gate.angles[1].to_radians()
            );
        }
        operation => panic!("Expected native RXY1Q gate, got {operation:?}"),
    }
}

#[test]
fn test_gate_expansion_cz() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        cz q[0], q[1];
    "#;

    let program = QASMParser::parse_str(qasm).unwrap();

    // The cz gate should be expanded to h; cx; h
    assert_eq!(program.operations.len(), 3);

    // Check first operation is h
    match &program.operations[0] {
        Operation::Gate { name, qubits, .. } => {
            assert_eq!(name, "H");
            assert_eq!(qubits, &[1]);
        }
        Operation::NativeGate(gate) => {
            assert_eq!(gate.gate_type, pecos_core::gate_type::GateType::H);
            assert_eq!(gate.qubits.len(), 1);
            assert_eq!({ gate.qubits[0].0 }, 1);
        }
        _ => panic!("Expected h gate"),
    }

    // Check second operation is cx
    match &program.operations[1] {
        Operation::Gate { name, qubits, .. } => {
            assert_eq!(name, "CX");
            assert_eq!(qubits, &[0, 1]);
        }
        Operation::NativeGate(gate) => {
            assert_eq!(gate.gate_type, pecos_core::gate_type::GateType::CX);
            assert_eq!(gate.qubits.len(), 2);
            assert_eq!({ gate.qubits[0].0 }, 0);
            assert_eq!({ gate.qubits[1].0 }, 1);
        }
        _ => panic!("Expected cx gate"),
    }

    // Check third operation is h
    match &program.operations[2] {
        Operation::Gate { name, qubits, .. } => {
            assert_eq!(name, "H");
            assert_eq!(qubits, &[1]);
        }
        Operation::NativeGate(gate) => {
            assert_eq!(gate.gate_type, pecos_core::gate_type::GateType::H);
            assert_eq!(gate.qubits.len(), 1);
            assert_eq!({ gate.qubits[0].0 }, 1);
        }
        _ => panic!("Expected h gate"),
    }
}

#[test]
fn test_gate_definitions_loaded() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
    "#;

    let program = QASMParser::parse_str(qasm).unwrap();

    // Check a known qelib1 gate exists in the definitions
    assert!(program.gate_definitions.contains_key("cx"));
    assert!(program.gate_definitions.contains_key("h"));
    assert!(program.gate_definitions.contains_key("x"));
    assert!(program.gate_definitions.contains_key("y"));
    assert!(program.gate_definitions.contains_key("z"));
}

#[test]
fn parse_with_config_honours_expand_gates_false() {
    // `parse_with_config` used to parse through `parse_str_raw`, which expands
    // unconditionally, so the flag only controlled a second, redundant pass.
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        gate mygate a { h a; }
        qreg q[2];
        creg c[2];
        mygate q[0];
        measure q -> c;
    "#;
    let unexpanded = QASMParser::parse_with_config(
        qasm,
        &ParseConfig {
            expand_gates: false,
            ..ParseConfig::default()
        },
    )
    .unwrap();
    assert!(
        matches!(&unexpanded.operations[0], Operation::Gate { name, .. } if name == "mygate"),
        "gate call must stay a Gate: {:?}",
        unexpanded.operations[0]
    );
    assert!(
        matches!(&unexpanded.operations[1], Operation::RegMeasure { .. }),
        "register measurement must stay a RegMeasure: {:?}",
        unexpanded.operations[1]
    );

    let expanded = QASMParser::parse_with_config(qasm, &ParseConfig::default()).unwrap();
    assert!(
        matches!(&expanded.operations[0], Operation::NativeGate(gate) if gate.gate_type == GateType::H),
        "gate call must expand to H: {:?}",
        expanded.operations[0]
    );
    assert!(
        expanded
            .operations
            .iter()
            .all(|op| !matches!(op, Operation::RegMeasure { .. })),
        "register measurement must expand per qubit"
    );
}

#[test]
fn one_expansion_call_lowers_everything_to_native_gates() {
    // qelib gates such as `sx` and `u3` used to need two `expand_gates` calls:
    // the first lowered them to canonical `Gate` nodes named `SX` and `U`, and
    // only a second call turned those into `NativeGate`s. One call must do it.
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        gate mygate a { h a; sx a; }
        qreg q[2];
        mygate q[0];
        u3(0.1, 0.2, 0.3) q[1];
        cx q[0], q[1];
    "#;
    let program = QASMParser::parse_str(qasm).unwrap();
    let leftover: Vec<_> = program
        .operations
        .iter()
        .filter(|op| matches!(op, Operation::Gate { .. }))
        .collect();
    assert!(
        leftover.is_empty(),
        "unlowered gate calls remain: {leftover:?}"
    );
    assert!(
        program
            .operations
            .iter()
            .all(|op| matches!(op, Operation::NativeGate(_))),
        "every operation must be native: {:?}",
        program.operations
    );
}

fn assert_native_gate(operation: &Operation, gate_type: GateType, qubits: &[usize]) {
    let Operation::NativeGate(gate) = operation else {
        panic!("Expected NativeGate({gate_type:?}), got {operation:?}");
    };
    assert_eq!(
        *gate,
        Gate::new(
            gate_type,
            vec![],
            vec![],
            qubits.iter().copied().map(QubitId).collect::<Vec<_>>(),
        )
    );
}

#[test]
fn gate_body_barrier_and_reset_lower_and_run() {
    let qasm = r"
        OPENQASM 2.0;
        gate g a { H a; barrier a; reset a; }
        qreg q[1];
        g q[0];
    ";
    let program = QASMParser::parse_str(qasm).unwrap();
    assert_eq!(program.operations.len(), 3);
    assert_native_gate(&program.operations[0], GateType::H, &[0]);
    assert!(matches!(&program.operations[1], Operation::Barrier { qubits } if qubits == &[0]));
    assert_native_gate(&program.operations[2], GateType::PZ, &[0]);

    let results = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .run(5)
        .unwrap();
    assert_eq!(results.len(), 5);
}

#[test]
fn native_alias_in_gate_body_lowers_without_reentering_definition() {
    let qasm = r"
        OPENQASM 2.0;
        gate SXXDG a,b { SXXdg a,b; }
        qreg q[2];
        SXXDG q[0],q[1];
    ";
    let program = QASMParser::parse_str(qasm).unwrap();
    assert_eq!(program.operations.len(), 1);
    assert_native_gate(&program.operations[0], GateType::SXXdg, &[0, 1]);

    let results = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .run(5)
        .unwrap();
    assert_eq!(results.len(), 5);
}

#[test]
fn nested_gate_bodies_lower_to_native_gates() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        gate inner a { h a; }
        gate outer a { inner a; x a; }
        qreg q[1];
        outer q[0];
    "#;
    let program = QASMParser::parse_str(qasm).unwrap();
    assert_eq!(program.operations.len(), 2);
    assert_native_gate(&program.operations[0], GateType::H, &[0]);
    assert_native_gate(&program.operations[1], GateType::X, &[0]);
}

#[test]
fn conditional_gate_body_barrier_and_reset_keep_condition() {
    let qasm = r"
        OPENQASM 2.0;
        gate g a { H a; barrier a; reset a; }
        qreg q[1];
        creg c[1];
        if (c == 1) g q[0];
    ";
    let program = QASMParser::parse_str(qasm).unwrap();
    assert_eq!(program.operations.len(), 3);
    let inner: Vec<_> = program
        .operations
        .iter()
        .map(|op| {
            let Operation::If {
                condition,
                operation,
            } = op
            else {
                panic!("Expected conditional operation, got {op:?}");
            };
            assert_eq!(condition.to_string(), "(c == 1)");
            operation.as_ref()
        })
        .collect();
    assert_native_gate(inner[0], GateType::H, &[0]);
    assert!(matches!(inner[1], Operation::Barrier { qubits } if qubits == &[0]));
    assert_native_gate(inner[2], GateType::PZ, &[0]);
}

#[test]
fn measure_in_gate_body_is_rejected_during_parsing() {
    // The gate-call syntax reaches lowering, where the missing classical mapping
    // must be rejected instead of leaving a string gate for the engine.
    let qasm = r"
        OPENQASM 2.0;
        gate g a { measure a; }
        qreg q[1];
        g q[0];
    ";
    let error = QASMParser::parse_str(qasm).unwrap_err();
    let PecosError::CompileInvalidOperation { operation, reason } = error else {
        panic!("Expected invalid operation, got {error}");
    };
    assert_eq!(operation, "QASM operation");
    assert_eq!(
        reason,
        "Measure operations require classical register mapping and should not appear in gate expansion"
    );
}
