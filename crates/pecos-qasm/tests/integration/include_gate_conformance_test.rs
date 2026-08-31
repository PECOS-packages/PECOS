// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! This is a PROJECTIVE check: each gate is compared to its reference up to one
//! shared global phase, because a boundary conversion is only required to be exact
//! up to an unobservable global phase. It therefore does NOT pin phase exactness --
//! `phase_exactness_test.rs` is the test that does. Every RELATIVE phase, including
//! the control-side phase that distinguishes a controlled rotation from a controlled
//! phase, is compared exactly.

use num_complex::Complex64;
use pecos_engines::{ClassicalEngine, DenseStateVecEngine, Engine};
use pecos_qasm::{QASMEngine, ast::GateDefinition};
use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_3, FRAC_PI_4, PI, TAU};
use std::fmt::Write;
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;

const TOLERANCE: f64 = 1e-9;

type Matrix = Vec<Vec<Complex64>>;

fn complex(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

fn cis(angle: f64) -> Complex64 {
    Complex64::from_polar(1.0, angle)
}

fn matrix<const N: usize>(rows: [[Complex64; N]; N]) -> Matrix {
    rows.into_iter().map(Vec::from).collect()
}

fn identity(dimension: usize) -> Matrix {
    let mut result = vec![vec![complex(0.0, 0.0); dimension]; dimension];
    for (index, row) in result.iter_mut().enumerate() {
        row[index] = complex(1.0, 0.0);
    }
    result
}

fn h_matrix() -> Matrix {
    matrix([
        [complex(FRAC_1_SQRT_2, 0.0), complex(FRAC_1_SQRT_2, 0.0)],
        [complex(FRAC_1_SQRT_2, 0.0), complex(-FRAC_1_SQRT_2, 0.0)],
    ])
}

fn x_matrix() -> Matrix {
    matrix([
        [complex(0.0, 0.0), complex(1.0, 0.0)],
        [complex(1.0, 0.0), complex(0.0, 0.0)],
    ])
}

fn y_matrix() -> Matrix {
    matrix([
        [complex(0.0, 0.0), complex(0.0, -1.0)],
        [complex(0.0, 1.0), complex(0.0, 0.0)],
    ])
}

fn z_matrix() -> Matrix {
    matrix([
        [complex(1.0, 0.0), complex(0.0, 0.0)],
        [complex(0.0, 0.0), complex(-1.0, 0.0)],
    ])
}

fn phase_matrix(angle: f64) -> Matrix {
    matrix([
        [complex(1.0, 0.0), complex(0.0, 0.0)],
        [complex(0.0, 0.0), cis(angle)],
    ])
}

fn rx_matrix(theta: f64) -> Matrix {
    let cosine = (theta / 2.0).cos();
    let sine = (theta / 2.0).sin();
    matrix([
        [complex(cosine, 0.0), complex(0.0, -sine)],
        [complex(0.0, -sine), complex(cosine, 0.0)],
    ])
}

fn ry_matrix(theta: f64) -> Matrix {
    let cosine = (theta / 2.0).cos();
    let sine = (theta / 2.0).sin();
    matrix([
        [complex(cosine, 0.0), complex(-sine, 0.0)],
        [complex(sine, 0.0), complex(cosine, 0.0)],
    ])
}

fn rz_matrix(theta: f64) -> Matrix {
    matrix([
        [cis(-theta / 2.0), complex(0.0, 0.0)],
        [complex(0.0, 0.0), cis(theta / 2.0)],
    ])
}

fn u_matrix(theta: f64, phi: f64, lambda: f64) -> Matrix {
    let cosine = (theta / 2.0).cos();
    let sine = (theta / 2.0).sin();
    matrix([
        [complex(cosine, 0.0), -cis(lambda) * sine],
        [cis(phi) * sine, cis(phi + lambda) * cosine],
    ])
}

fn rxy_matrix(theta: f64, phi: f64) -> Matrix {
    let cosine = (theta / 2.0).cos();
    let sine = (theta / 2.0).sin();
    let minus_i = complex(0.0, -1.0);
    matrix([
        [complex(cosine, 0.0), minus_i * cis(-phi) * sine],
        [minus_i * cis(phi) * sine, complex(cosine, 0.0)],
    ])
}

fn sx_matrix() -> Matrix {
    matrix([
        [complex(0.5, 0.5), complex(0.5, -0.5)],
        [complex(0.5, -0.5), complex(0.5, 0.5)],
    ])
}

fn sxdg_matrix() -> Matrix {
    matrix([
        [complex(0.5, -0.5), complex(0.5, 0.5)],
        [complex(0.5, 0.5), complex(0.5, -0.5)],
    ])
}

fn controlled(target: &Matrix) -> Matrix {
    let mut result = identity(4);
    for target_row in 0..2 {
        for target_column in 0..2 {
            result[2 + target_row][2 + target_column] = target[target_row][target_column];
        }
    }
    result
}

fn swap_matrix() -> Matrix {
    matrix([
        [
            complex(1.0, 0.0),
            complex(0.0, 0.0),
            complex(0.0, 0.0),
            complex(0.0, 0.0),
        ],
        [
            complex(0.0, 0.0),
            complex(0.0, 0.0),
            complex(1.0, 0.0),
            complex(0.0, 0.0),
        ],
        [
            complex(0.0, 0.0),
            complex(1.0, 0.0),
            complex(0.0, 0.0),
            complex(0.0, 0.0),
        ],
        [
            complex(0.0, 0.0),
            complex(0.0, 0.0),
            complex(0.0, 0.0),
            complex(1.0, 0.0),
        ],
    ])
}

fn rzz_matrix(theta: f64) -> Matrix {
    let same = cis(-theta / 2.0);
    let different = cis(theta / 2.0);
    matrix([
        [
            same,
            complex(0.0, 0.0),
            complex(0.0, 0.0),
            complex(0.0, 0.0),
        ],
        [
            complex(0.0, 0.0),
            different,
            complex(0.0, 0.0),
            complex(0.0, 0.0),
        ],
        [
            complex(0.0, 0.0),
            complex(0.0, 0.0),
            different,
            complex(0.0, 0.0),
        ],
        [
            complex(0.0, 0.0),
            complex(0.0, 0.0),
            complex(0.0, 0.0),
            same,
        ],
    ])
}

fn rxx_matrix(theta: f64) -> Matrix {
    let cosine = complex((theta / 2.0).cos(), 0.0);
    let coupling = complex(0.0, -(theta / 2.0).sin());
    matrix([
        [cosine, complex(0.0, 0.0), complex(0.0, 0.0), coupling],
        [complex(0.0, 0.0), cosine, coupling, complex(0.0, 0.0)],
        [complex(0.0, 0.0), coupling, cosine, complex(0.0, 0.0)],
        [coupling, complex(0.0, 0.0), complex(0.0, 0.0), cosine],
    ])
}


/// Conventional two-qubit root: `((1+i) I + (1-i) P) / 2` for an involution P.
fn conventional_root(pauli: &Matrix) -> Matrix {
    let a = complex(0.5, 0.5);
    let b = complex(0.5, -0.5);
    (0..4)
        .map(|r| {
            (0..4)
                .map(|c| {
                    let ident = if r == c { complex(1.0, 0.0) } else { complex(0.0, 0.0) };
                    a * ident + b * pauli[r][c]
                })
                .collect()
        })
        .collect()
}

/// Adjoint of the conventional root: conjugate transpose (entries are symmetric).
fn conventional_root_dagger(pauli: &Matrix) -> Matrix {
    conventional_root(pauli)
        .iter()
        .map(|row| row.iter().map(num_complex::Complex::conj).collect())
        .collect()
}

fn xx_pauli() -> Matrix {
    let o = complex(0.0, 0.0);
    let l = complex(1.0, 0.0);
    matrix([[o, o, o, l], [o, o, l, o], [o, l, o, o], [l, o, o, o]])
}

fn yy_pauli() -> Matrix {
    let o = complex(0.0, 0.0);
    let l = complex(1.0, 0.0);
    let m = complex(-1.0, 0.0);
    matrix([[o, o, o, m], [o, o, l, o], [o, l, o, o], [m, o, o, o]])
}

fn zz_pauli() -> Matrix {
    let o = complex(0.0, 0.0);
    let l = complex(1.0, 0.0);
    let m = complex(-1.0, 0.0);
    matrix([[l, o, o, o], [o, m, o, o], [o, o, m, o], [o, o, o, l]])
}

fn szz_matrix() -> Matrix {
    matrix([
        [
            complex(1.0, 0.0),
            complex(0.0, 0.0),
            complex(0.0, 0.0),
            complex(0.0, 0.0),
        ],
        [
            complex(0.0, 0.0),
            complex(0.0, 1.0),
            complex(0.0, 0.0),
            complex(0.0, 0.0),
        ],
        [
            complex(0.0, 0.0),
            complex(0.0, 0.0),
            complex(0.0, 1.0),
            complex(0.0, 0.0),
        ],
        [
            complex(0.0, 0.0),
            complex(0.0, 0.0),
            complex(0.0, 0.0),
            complex(1.0, 0.0),
        ],
    ])
}

fn toffoli_matrix() -> Matrix {
    let mut result = identity(8);
    result[6][6] = complex(0.0, 0.0);
    result[7][7] = complex(0.0, 0.0);
    result[6][7] = complex(1.0, 0.0);
    result[7][6] = complex(1.0, 0.0);
    result
}

fn reference_matrix(name: &str, parameters: &[f64]) -> Matrix {
    match name {
        "h" => h_matrix(),
        "x" => x_matrix(),
        "y" => y_matrix(),
        "z" => z_matrix(),
        "id" => identity(2),
        "s" => phase_matrix(PI / 2.0),
        "sdg" | "Sdg" => phase_matrix(-PI / 2.0),
        "t" => phase_matrix(FRAC_PI_4),
        "tdg" | "Tdg" => phase_matrix(-FRAC_PI_4),
        "sx" => sx_matrix(),
        "sxdg" => sxdg_matrix(),
        "rx" | "RX" => rx_matrix(parameters[0]),
        "ry" | "RY" => ry_matrix(parameters[0]),
        "rz" | "Rz" => rz_matrix(parameters[0]),
        "phase" | "p" | "u1" => phase_matrix(parameters[0]),
        "u" | "u3" => u_matrix(parameters[0], parameters[1], parameters[2]),
        "u2" => u_matrix(PI / 2.0, parameters[0], parameters[1]),
        "U1q" | "rxy1q" | "r1xy" => rxy_matrix(parameters[0], parameters[1]),
        "cx" | "cnot" | "CNOT" => controlled(&x_matrix()),
        "cy" => controlled(&y_matrix()),
        "cz" | "cphase180" => controlled(&z_matrix()),
        "swap" => swap_matrix(),
        "csx" => controlled(&sx_matrix()),
        "crz" => controlled(&rz_matrix(parameters[0])),
        "crx" => controlled(&rx_matrix(parameters[0])),
        "cry" => controlled(&ry_matrix(parameters[0])),
        "cphase" | "cu1" | "cp" => controlled(&phase_matrix(parameters[0])),
        "cphase90" => controlled(&phase_matrix(PI / 2.0)),
        "rzz" => rzz_matrix(parameters[0]),
        "rxx" => rxx_matrix(parameters[0]),
        "szz" | "ZZ" => szz_matrix(),
        "szzdg" => conventional_root_dagger(&zz_pauli()),
        "sxx" => conventional_root(&xx_pauli()),
        "sxxdg" => conventional_root_dagger(&xx_pauli()),
        "syy" => conventional_root(&yy_pauli()),
        "syydg" => conventional_root_dagger(&yy_pauli()),
        "ccx" => toffoli_matrix(),
        _ => panic!("gate '{name}' has no textbook reference"),
    }
}

fn has_non_periodic_theta(name: &str) -> bool {
    matches!(
        name,
        "rx" | "RX"
            | "ry"
            | "RY"
            | "rz"
            | "Rz"
            | "crx"
            | "cry"
            | "crz"
            | "rxx"
            | "rzz"
            | "u"
            | "u3"
            | "U1q"
            | "rxy1q"
            | "r1xy"
    )
}

fn parameter_cases(definition: &GateDefinition) -> Vec<Vec<f64>> {
    if definition.params.is_empty() {
        return vec![Vec::new()];
    }

    let baseline = [0.23, -0.41, 0.59];
    let mut cases = Vec::new();
    for parameter_index in 0..definition.params.len() {
        for probe in [0.7, FRAC_PI_3] {
            let mut parameters = baseline[..definition.params.len()].to_vec();
            parameters[parameter_index] = probe;
            cases.push(parameters);
        }
    }

    if has_non_periodic_theta(&definition.name) {
        for probe in [TAU, 3.0 * PI] {
            let mut parameters = baseline[..definition.params.len()].to_vec();
            parameters[0] = probe;
            cases.push(parameters);
        }
    }
    cases
}

fn gate_invocation(definition: &GateDefinition, parameters: &[f64]) -> String {
    let mut invocation = definition.name.clone();
    if !parameters.is_empty() {
        invocation.push('(');
        for (index, parameter) in parameters.iter().enumerate() {
            if index > 0 {
                invocation.push_str(", ");
            }
            write!(invocation, "{parameter:.17}").expect("writing to String cannot fail");
        }
        invocation.push(')');
    }
    invocation.push(' ');
    for index in 0..definition.qargs.len() {
        if index > 0 {
            invocation.push_str(", ");
        }
        write!(invocation, "q[{index}]").expect("writing to String cannot fail");
    }
    invocation.push(';');
    invocation
}

fn compile_gate(
    include_name: &str,
    definition: &GateDefinition,
    parameters: &[f64],
) -> Vec<pecos_engines::ByteMessage> {
    let invocation = gate_invocation(definition, parameters);
    let qasm = format!(
        "OPENQASM 2.0;\ninclude \"{include_name}\";\nqreg q[{}];\n{invocation}",
        definition.qargs.len()
    );
    let mut engine = QASMEngine::from_str(&qasm).unwrap_or_else(|error| {
        panic!(
            "{include_name} {}{parameters:?} failed to parse: {error}",
            definition.name
        )
    });
    let mut messages = Vec::new();
    loop {
        let message = engine.generate_commands().unwrap_or_else(|error| {
            panic!(
                "{include_name} {}{parameters:?} failed to lower: {error}",
                definition.name
            )
        });
        let native_gates = message.quantum_ops().unwrap_or_else(|error| {
            panic!(
                "{include_name} {}{parameters:?} produced invalid native gates: {error}",
                definition.name
            )
        });
        if native_gates.is_empty() {
            break;
        }
        messages.push(message);
    }
    messages
}

fn logical_to_simulator_basis(logical_basis: usize, qubit_count: usize) -> usize {
    let mut simulator_basis = 0;
    for logical_qubit in 0..qubit_count {
        let logical_bit = (logical_basis >> (qubit_count - logical_qubit - 1)) & 1;
        simulator_basis |= logical_bit << logical_qubit;
    }
    simulator_basis
}

fn executed_matrix(include_name: &str, definition: &GateDefinition, parameters: &[f64]) -> Matrix {
    let messages = compile_gate(include_name, definition, parameters);
    let qubit_count = definition.qargs.len();
    let dimension = 1 << qubit_count;
    let mut result = vec![vec![complex(0.0, 0.0); dimension]; dimension];

    for logical_input in 0..dimension {
        let simulator_input = logical_to_simulator_basis(logical_input, qubit_count);
        let mut executor = DenseStateVecEngine::new(qubit_count);
        if simulator_input != 0 {
            executor.simulator_mut().set_amplitude(0, complex(0.0, 0.0));
            executor
                .simulator_mut()
                .set_amplitude(simulator_input, complex(1.0, 0.0));
        }
        for message in &messages {
            executor.process(message.clone()).unwrap_or_else(|error| {
                panic!(
                    "{include_name} {}{parameters:?} failed to execute: {error}",
                    definition.name
                )
            });
        }
        let state = executor.simulator_mut().state();
        for (logical_output, row) in result.iter_mut().enumerate() {
            let simulator_output = logical_to_simulator_basis(logical_output, qubit_count);
            row[logical_input] = state[simulator_output];
        }
    }
    result
}

fn assert_matrix_matches(
    include_name: &str,
    gate_name: &str,
    parameters: &[f64],
    actual: &Matrix,
    expected: &Matrix,
) {
    assert_eq!(actual.len(), expected.len());
    let (pivot_row, pivot_column, _) = expected
        .iter()
        .enumerate()
        .flat_map(|(row, entries)| {
            entries
                .iter()
                .enumerate()
                .map(move |(column, value)| (row, column, value.norm()))
        })
        .max_by(|left, right| left.2.total_cmp(&right.2))
        .expect("a unitary matrix is nonempty");
    let phase_ratio = actual[pivot_row][pivot_column] / expected[pivot_row][pivot_column];
    let phase = phase_ratio / phase_ratio.norm();

    for (row_index, (actual_row, expected_row)) in actual.iter().zip(expected).enumerate() {
        for (column_index, (&actual_entry, &expected_entry)) in
            actual_row.iter().zip(expected_row).enumerate()
        {
            let phase_adjusted = actual_entry / phase;
            let error = (phase_adjusted - expected_entry).norm();
            assert!(
                error <= TOLERANCE,
                "{include_name} {gate_name}{parameters:?} differs at [{row_index},{column_index}]: \
                 actual={actual_entry}, shared_phase={phase}, adjusted={phase_adjusted}, \
                 expected={expected_entry}, error={error}"
            );
        }
    }
}

fn include_paths() -> Vec<PathBuf> {
    let include_directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("includes");
    let mut paths: Vec<_> = fs::read_dir(&include_directory)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", include_directory.display()))
        .map(|entry| {
            entry
                .expect("include directory entry must be readable")
                .path()
        })
        .filter(|path| path.extension().is_some_and(|extension| extension == "inc"))
        .collect();
    paths.sort();
    paths
}

fn gate_definitions(include_name: &str) -> Vec<GateDefinition> {
    let qasm = format!("OPENQASM 2.0;\ninclude \"{include_name}\";\nqreg q[1];");
    let engine = QASMEngine::from_str(&qasm)
        .unwrap_or_else(|error| panic!("failed to parse {include_name}: {error}"));
    engine
        .gate_definitions()
        .expect("the engine has a loaded program")
        .values()
        .cloned()
        .collect()
}

#[test]
fn every_include_gate_matches_its_textbook_unitary() {
    let paths = include_paths();
    assert!(
        !paths.is_empty(),
        "the standard includes directory is empty"
    );

    for path in paths {
        let include_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .expect("include filename must be valid UTF-8");
        let definitions = gate_definitions(include_name);
        assert!(!definitions.is_empty(), "{include_name} defines no gates");

        for definition in definitions {
            for parameters in parameter_cases(&definition) {
                let expected = reference_matrix(&definition.name, &parameters);
                let actual = executed_matrix(include_name, &definition, &parameters);
                assert_matrix_matches(
                    include_name,
                    &definition.name,
                    &parameters,
                    &actual,
                    &expected,
                );
            }
        }
    }
}
