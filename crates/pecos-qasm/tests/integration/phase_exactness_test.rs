use num_complex::Complex64;
use pecos_core::prelude::GateType;
use pecos_core::{Angle64, QubitId};
use pecos_qasm::{Operation, QASMParser};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, StateVecSoA32};

const TOLERANCE: f64 = 3e-6;

fn complex64(amplitude: num_complex::Complex32) -> Complex64 {
    Complex64::new(f64::from(amplitude.re), f64::from(amplitude.im))
}

fn native_u_angles(include: &str, spelling: &str, lambda: &str) -> [Angle64; 3] {
    let qasm =
        format!("OPENQASM 2.0;\ninclude \"{include}\";\nqreg q[1];\n{spelling}({lambda}) q[0];");
    let program = QASMParser::parse_str(&qasm).expect("phase gate must parse");
    assert_eq!(program.operations.len(), 1);

    match &program.operations[0] {
        Operation::Gate {
            name, parameters, ..
        } => {
            assert_eq!(name, "U", "{spelling} must lower to native U");
            assert_eq!(parameters.len(), 3);
            parameters
                .iter()
                .map(|&angle| Angle64::from_radians(angle))
                .collect::<Vec<_>>()
                .try_into()
                .expect("U has three angles")
        }
        Operation::NativeGate(gate) => {
            assert_eq!(gate.gate_type, GateType::U, "{spelling} must lower to U");
            gate.angles
                .iter()
                .copied()
                .collect::<Vec<_>>()
                .try_into()
                .expect("U has three angles")
        }
        operation => panic!("{spelling} lowered to unexpected operation {operation:?}"),
    }
}

fn assert_qasm_p_matrix(lambda: &str, expected_high: Complex64) {
    for include in ["qelib1.inc", "hqslib1.inc"] {
        let [theta, phi, lambda] = native_u_angles(include, "p", lambda);
        assert_eq!(theta, Angle64::ZERO);
        assert_eq!(phi, Angle64::ZERO);

        let mut zero = StateVecSoA32::new(1);
        zero.u(theta, phi, lambda, &[QubitId(0)]);
        let zero_column = [zero.get_amplitude(0), zero.get_amplitude(1)];
        assert!((complex64(zero_column[0]) - Complex64::new(1.0, 0.0)).norm() < TOLERANCE);
        assert!(complex64(zero_column[1]).norm() < TOLERANCE);

        let mut one = StateVecSoA32::new(1);
        one.x(&[QubitId(0)]);
        one.u(theta, phi, lambda, &[QubitId(0)]);
        let one_column = [one.get_amplitude(0), one.get_amplitude(1)];
        assert!(complex64(one_column[0]).norm() < TOLERANCE);
        assert!((complex64(one_column[1]) - expected_high).norm() < TOLERANCE);
    }
}

#[test]
fn qasm_p_zero_is_exact_identity() {
    assert_qasm_p_matrix("0", Complex64::new(1.0, 0.0));
}

#[test]
fn qasm_p_pi_over_4_is_exact_t() {
    assert_qasm_p_matrix(
        "pi/4",
        Complex64::new(
            std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2,
        ),
    );
}

#[test]
fn qasm_p_pi_over_2_is_exact_sz() {
    assert_qasm_p_matrix("pi/2", Complex64::new(0.0, 1.0));
}

#[test]
fn qasm_p_pi_is_exact_z() {
    assert_qasm_p_matrix("pi", Complex64::new(-1.0, 0.0));
}

fn apply_expanded_operation(sim: &mut StateVecSoA32, operation: &Operation) {
    match operation {
        Operation::Gate {
            name,
            parameters,
            qubits,
        } if name == "U" => {
            assert_eq!(parameters.len(), 3);
            sim.u(
                Angle64::from_radians(parameters[0]),
                Angle64::from_radians(parameters[1]),
                Angle64::from_radians(parameters[2]),
                &[QubitId(qubits[0])],
            );
        }
        Operation::Gate { name, qubits, .. } if name == "CX" => {
            sim.cx(&[(QubitId(qubits[0]), QubitId(qubits[1]))]);
        }
        Operation::Gate { name, qubits, .. } if name == "H" => {
            sim.h(&[QubitId(qubits[0])]);
        }
        Operation::Gate {
            name,
            parameters,
            qubits,
        } if name == "RX" => {
            sim.rx(Angle64::from_radians(parameters[0]), &[QubitId(qubits[0])]);
        }
        Operation::Gate {
            name,
            parameters,
            qubits,
        } if name == "RZ" => {
            sim.rz(Angle64::from_radians(parameters[0]), &[QubitId(qubits[0])]);
        }
        Operation::Gate {
            name,
            parameters,
            qubits,
        } if name == "RXY1Q" => {
            sim.rxy1q(
                Angle64::from_radians(parameters[0]),
                Angle64::from_radians(parameters[1]),
                &[QubitId(qubits[0])],
            );
        }
        Operation::NativeGate(gate) if gate.gate_type == GateType::U => {
            sim.u(gate.angles[0], gate.angles[1], gate.angles[2], &gate.qubits);
        }
        Operation::NativeGate(gate) if gate.gate_type == GateType::CX => {
            sim.cx(&[(gate.qubits[0], gate.qubits[1])]);
        }
        Operation::NativeGate(gate) if gate.gate_type == GateType::H => {
            sim.h(&gate.qubits);
        }
        Operation::NativeGate(gate) if gate.gate_type == GateType::RX => {
            sim.rx(gate.angles[0], &gate.qubits);
        }
        Operation::NativeGate(gate) if gate.gate_type == GateType::RZ => {
            sim.rz(gate.angles[0], &gate.qubits);
        }
        Operation::NativeGate(gate) if gate.gate_type == GateType::RXY1Q => {
            sim.rxy1q(gate.angles[0], gate.angles[1], &gate.qubits);
        }
        operation => panic!("unexpected expanded operation {operation:?}"),
    }
}

fn single_qubit_columns(invocation: &str) -> [[Complex64; 2]; 2] {
    let qasm = format!("OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];\n{invocation} q[0];");
    let program = QASMParser::parse_str(&qasm).expect("single-qubit gate must parse");
    let mut columns = [[Complex64::new(0.0, 0.0); 2]; 2];

    for (basis, column) in columns.iter_mut().enumerate() {
        let mut sim = StateVecSoA32::new(1);
        if basis == 1 {
            sim.x(&[QubitId(0)]);
        }
        for operation in &program.operations {
            apply_expanded_operation(&mut sim, operation);
        }
        *column = [
            complex64(sim.get_amplitude(0)),
            complex64(sim.get_amplitude(1)),
        ];
    }

    columns
}

fn assert_columns_equal(label: &str, actual: &[[Complex64; 2]; 2], expected: &[[Complex64; 2]; 2]) {
    for (column, (actual_column, expected_column)) in actual.iter().zip(expected).enumerate() {
        for (row, (actual, expected)) in actual_column.iter().zip(expected_column).enumerate() {
            let error = (actual - expected).norm();
            assert!(
                error < TOLERANCE,
                "{label}, column {column}, row {row}: actual={actual}, expected={expected}, error={error:e}"
            );
        }
    }
}

#[test]
fn qelib1_u3_zero_theta_matches_u1_phase_exactly() {
    let u3 = single_qubit_columns("u3(0, 0, 0.7)");
    let u1 = single_qubit_columns("u1(0.7)");
    assert_columns_equal("u3(0,0,lambda) versus u1(lambda)", &u3, &u1);
}

#[test]
fn qelib1_u2_matches_native_u_phase_exactly() {
    let u2 = single_qubit_columns("u2(0.4, 0.8)");
    let native_u = single_qubit_columns("U(pi/2, 0.4, 0.8)");
    assert_columns_equal("u2(phi,lambda) versus U(pi/2,phi,lambda)", &u2, &native_u);
}

fn assert_controlled_phase(include: &str, spelling: &str) {
    let qasm =
        format!("OPENQASM 2.0;\ninclude \"{include}\";\nqreg q[2];\n{spelling}(pi/3) q[0],q[1];");
    let program = QASMParser::parse_str(&qasm).expect("controlled phase gate must parse");
    let expected_phase = Complex64::from_polar(1.0, std::f64::consts::PI / 3.0);

    for basis in 0..4 {
        let mut sim = StateVecSoA32::new(2);
        for qubit in 0..2 {
            if basis & (1 << qubit) != 0 {
                sim.x(&[QubitId(qubit)]);
            }
        }
        for operation in &program.operations {
            apply_expanded_operation(&mut sim, operation);
        }

        for index in 0..4 {
            let actual = complex64(sim.get_amplitude(index));
            let expected = if index == basis {
                if basis == 3 {
                    expected_phase
                } else {
                    Complex64::new(1.0, 0.0)
                }
            } else {
                Complex64::new(0.0, 0.0)
            };
            assert!(
                (actual - expected).norm() < TOLERANCE,
                "{include} {spelling}, input {basis}, amplitude {index}: actual={actual}, expected={expected}"
            );
        }
    }
}

#[test]
fn controlled_phase_family_remains_exact() {
    assert_controlled_phase("hqslib1.inc", "cp");
    assert_controlled_phase("qelib1.inc", "cu1");
    assert_controlled_phase("qelib1.inc", "cphase");
}

#[test]
fn phase_and_u1_aliases_lower_to_exact_u() {
    for include in ["qelib1.inc", "hqslib1.inc"] {
        for spelling in ["phase", "u1"] {
            let [theta, phi, lambda] = native_u_angles(include, spelling, "pi/3");
            assert_eq!(theta, Angle64::ZERO);
            assert_eq!(phi, Angle64::ZERO);
            assert!(lambda.abs_diff_eq_radians(
                &Angle64::from_radians(std::f64::consts::PI / 3.0),
                1e-14,
            ));
        }
    }
}
