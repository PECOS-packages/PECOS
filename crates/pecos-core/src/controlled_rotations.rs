// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Exact lowering of controlled-rotation boundary spellings.

use crate::{Angle64, Gate, QubitId};

/// Lower `CRZ(theta)` to native rotations.
///
/// The identity is
/// `CRZ(theta) = (I (x) RZ(theta/2)) . RZZ(-theta/2)`, with no global phase
/// before angle reduction. The stored lowering is exact up to a global phase of
/// ±1 that appears when a halved angle crosses the 2π reduction.
/// `theta` is supplied in radians and is halved as an `f64` before either half
/// is normalized into [`Angle64`]. This order is required because `CRZ` is
/// 4π-periodic while a stored `Angle64` has already reduced its input modulo 2π.
#[must_use]
pub fn lower_crz(theta_radians: f64, control: QubitId, target: QubitId) -> [Gate; 2] {
    let half_theta = theta_radians / 2.0;
    [
        Gate::rzz(Angle64::from_radians(-half_theta), &[(control, target)]),
        Gate::rz(Angle64::from_radians(half_theta), &[target]),
    ]
}

/// Lower `CRX(theta) = (I (x) H) . CRZ(theta) . (I (x) H)`.
#[must_use]
pub fn lower_crx(theta_radians: f64, control: QubitId, target: QubitId) -> [Gate; 4] {
    let [rzz, rz] = lower_crz(theta_radians, control, target);
    [Gate::h(&[target]), rzz, rz, Gate::h(&[target])]
}

/// Lower `CRY(theta) = (I (x) SXdg) . CRZ(theta) . (I (x) SX)`.
///
/// Circuit emission order is `SX`, the `CRZ` lowering, then `SXdg`.
#[must_use]
pub fn lower_cry(theta_radians: f64, control: QubitId, target: QubitId) -> [Gate; 4] {
    let [rzz, rz] = lower_crz(theta_radians, control, target);
    [Gate::sx(&[target]), rzz, rz, Gate::sxdg(&[target])]
}

/// Lower controlled phase while retaining its relative phase.
///
/// The identity is
/// `CPhase(lambda) = (U(0,0,lambda/2) (x) RZ(lambda/2)) . RZZ(-lambda/2)`.
/// `U(0,0,lambda/2)` carries the `exp(i lambda/4)` factor. The stored lowering
/// is exact up to a further global phase of ±1 when a halved angle crosses the
/// 2π reduction.
#[must_use]
pub fn lower_cphase(lambda_radians: f64, control: QubitId, target: QubitId) -> [Gate; 3] {
    let half_lambda = lambda_radians / 2.0;
    let half_angle = Angle64::from_radians(half_lambda);
    [
        Gate::rzz(Angle64::from_radians(-half_lambda), &[(control, target)]),
        Gate::u(Angle64::ZERO, Angle64::ZERO, half_angle, &[control]),
        Gate::rz(half_angle, &[target]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate_type::GateType;
    use num_complex::Complex64;

    const TOLERANCE: f64 = 1.0e-12;

    fn apply_single(state: &mut [Complex64; 4], qubit: usize, matrix: [[Complex64; 2]; 2]) {
        let mask = 1usize << (1 - qubit);
        for base in 0..4 {
            if base & mask == 0 {
                let paired = base | mask;
                let zero = state[base];
                let one = state[paired];
                state[base] = matrix[0][0] * zero + matrix[0][1] * one;
                state[paired] = matrix[1][0] * zero + matrix[1][1] * one;
            }
        }
    }

    fn apply_gate(state: &mut [Complex64; 4], gate: &Gate) {
        let i = Complex64::new(0.0, 1.0);
        match gate.gate_type {
            GateType::H => {
                let s = std::f64::consts::FRAC_1_SQRT_2;
                apply_single(
                    state,
                    gate.qubits[0].index(),
                    [
                        [Complex64::new(s, 0.0), Complex64::new(s, 0.0)],
                        [Complex64::new(s, 0.0), Complex64::new(-s, 0.0)],
                    ],
                );
            }
            GateType::SX | GateType::SXdg => {
                let sign = if gate.gate_type == GateType::SX {
                    1.0
                } else {
                    -1.0
                };
                let diagonal = Complex64::new(0.5, 0.5 * sign);
                let off_diagonal = Complex64::new(0.5, -0.5 * sign);
                apply_single(
                    state,
                    gate.qubits[0].index(),
                    [[diagonal, off_diagonal], [off_diagonal, diagonal]],
                );
            }
            GateType::RZ => {
                let theta = gate.angles[0].to_radians_signed();
                apply_single(
                    state,
                    gate.qubits[0].index(),
                    [
                        [(-i * theta / 2.0).exp(), Complex64::new(0.0, 0.0)],
                        [Complex64::new(0.0, 0.0), (i * theta / 2.0).exp()],
                    ],
                );
            }
            GateType::RZZ => {
                let theta = gate.angles[0].to_radians_signed();
                for (basis, amplitude) in state.iter_mut().enumerate() {
                    let parity = ((basis >> 1) ^ basis) & 1;
                    let eigenvalue = if parity == 0 { 1.0 } else { -1.0 };
                    *amplitude *= (-i * theta * eigenvalue / 2.0).exp();
                }
            }
            GateType::U => {
                let lambda = gate.angles[2].to_radians_signed();
                apply_single(
                    state,
                    gate.qubits[0].index(),
                    [
                        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                        [Complex64::new(0.0, 0.0), (i * lambda).exp()],
                    ],
                );
            }
            other => panic!("unexpected lowered gate {other:?}"),
        }
    }

    fn matrix_from_lowering(gates: &[Gate]) -> [[Complex64; 4]; 4] {
        std::array::from_fn(|column| {
            let mut state = [Complex64::new(0.0, 0.0); 4];
            state[column] = Complex64::new(1.0, 0.0);
            for gate in gates {
                apply_gate(&mut state, gate);
            }
            state
        })
    }

    fn assert_matrix_eq_up_to_one_global_phase(
        actual: [[Complex64; 4]; 4],
        expected: [[Complex64; 4]; 4],
        angle: f64,
    ) {
        let (phase_column, phase_row, reference) = (0..4)
            .flat_map(|column| (0..4).map(move |row| (column, row)))
            .map(|(column, row)| (column, row, expected[column][row]))
            .max_by(|left, right| left.2.norm().total_cmp(&right.2.norm()))
            .expect("matrix has entries");
        let phase = actual[phase_column][phase_row] / reference;
        assert!((phase.norm() - 1.0).abs() < TOLERANCE);
        let crosses_reduction = ![
            -std::f64::consts::PI,
            std::f64::consts::PI / 3.0,
            std::f64::consts::PI,
        ]
        .iter()
        .any(|expected| (angle - expected).abs() < f64::EPSILON);
        if crosses_reduction {
            assert!(
                (phase - Complex64::new(1.0, 0.0)).norm() < TOLERANCE
                    || (phase + Complex64::new(1.0, 0.0)).norm() < TOLERANCE
            );
        } else {
            assert!((phase - Complex64::new(1.0, 0.0)).norm() < TOLERANCE);
        }
        for column in 0..4 {
            for row in 0..4 {
                assert!(
                    (actual[column][row] / phase - expected[column][row]).norm() < TOLERANCE,
                    "angle {angle}, column {column}, row {row}: actual={}, expected={}",
                    actual[column][row],
                    expected[column][row]
                );
            }
        }
    }

    fn controlled_reference(axis: char, theta: f64) -> [[Complex64; 4]; 4] {
        let mut matrix = [[Complex64::new(0.0, 0.0); 4]; 4];
        matrix[0][0] = Complex64::new(1.0, 0.0);
        matrix[1][1] = Complex64::new(1.0, 0.0);
        let c = Complex64::new((theta / 2.0).cos(), 0.0);
        let s = (theta / 2.0).sin();
        match axis {
            'X' => {
                matrix[2][2] = c;
                matrix[2][3] = Complex64::new(0.0, -s);
                matrix[3][2] = Complex64::new(0.0, -s);
                matrix[3][3] = c;
            }
            'Y' => {
                matrix[2][2] = c;
                matrix[2][3] = Complex64::new(s, 0.0);
                matrix[3][2] = Complex64::new(-s, 0.0);
                matrix[3][3] = c;
            }
            'Z' => {
                matrix[2][2] = Complex64::from_polar(1.0, -theta / 2.0);
                matrix[3][3] = Complex64::from_polar(1.0, theta / 2.0);
            }
            _ => unreachable!(),
        }
        matrix
    }

    #[test]
    fn controlled_rotation_lowerings_match_all_basis_columns() {
        for theta in [
            -std::f64::consts::PI,
            std::f64::consts::PI / 3.0,
            std::f64::consts::PI,
            std::f64::consts::TAU,
            3.0 * std::f64::consts::PI,
        ] {
            assert_matrix_eq_up_to_one_global_phase(
                matrix_from_lowering(&lower_crz(theta, QubitId(0), QubitId(1))),
                controlled_reference('Z', theta),
                theta,
            );
            assert_matrix_eq_up_to_one_global_phase(
                matrix_from_lowering(&lower_crx(theta, QubitId(0), QubitId(1))),
                controlled_reference('X', theta),
                theta,
            );
            assert_matrix_eq_up_to_one_global_phase(
                matrix_from_lowering(&lower_cry(theta, QubitId(0), QubitId(1))),
                controlled_reference('Y', theta),
                theta,
            );
        }
    }

    #[test]
    fn controlled_phase_lowering_preserves_phase() {
        for lambda in [
            -std::f64::consts::PI,
            std::f64::consts::PI / 3.0,
            std::f64::consts::PI,
            std::f64::consts::TAU,
            3.0 * std::f64::consts::PI,
        ] {
            let mut expected = [[Complex64::new(0.0, 0.0); 4]; 4];
            for (basis, row) in expected.iter_mut().enumerate().take(3) {
                row[basis] = Complex64::new(1.0, 0.0);
            }
            expected[3][3] = Complex64::from_polar(1.0, lambda);
            assert_matrix_eq_up_to_one_global_phase(
                matrix_from_lowering(&lower_cphase(lambda, QubitId(0), QubitId(1))),
                expected,
                lambda,
            );
        }
    }
}
