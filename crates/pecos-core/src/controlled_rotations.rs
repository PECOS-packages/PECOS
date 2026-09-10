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

use crate::{Angle64, Gate, PhaseGateError, QubitId};
use smallvec::{SmallVec, smallvec};
use std::f64::consts::{PI, TAU};

/// Return half the principal value in `(-π, π]` and the parity of removed turns.
/// Keep both reduction and parity in f64: storing the remainder in `Angle64`
/// would quantize it before the turn count is recovered.
fn reduced_half_and_wrap_parity(theta_radians: f64) -> (f64, bool) {
    let reduced = theta_radians.rem_euclid(TAU);
    let signed = if reduced > PI { reduced - TAU } else { reduced };
    let h = signed / 2.0;
    let wraps = ((theta_radians - signed) / TAU).round();
    // The rounded count has exact integer parity; compare exactly, without a tolerance.
    let needs_z = wraps.rem_euclid(2.0).total_cmp(&1.0).is_eq();
    (h, needs_z)
}

/// Lower `CRZ(theta)` to native rotations.
///
/// The identity is
/// `CRZ(theta) = (I (x) RZ(theta/2)) . RZZ(-theta/2)`, with no global phase
/// before angle reduction. Reduce `theta` to `(-π, π]` before halving so neither
/// stored leg reaches the ambiguous ±π pair. Each removed 2π turn contributes
/// a `Z` on the control; retaining its parity preserves the exact global phase
/// and 4π periodicity without quantizing the source angle through [`Angle64`].
#[must_use]
pub fn lower_crz(theta_radians: f64, control: QubitId, target: QubitId) -> SmallVec<[Gate; 3]> {
    let (half_theta, needs_z) = reduced_half_and_wrap_parity(theta_radians);
    let mut gates = SmallVec::new();
    if needs_z {
        gates.push(Gate::z(&[control]));
    }
    gates.extend([
        Gate::rzz(Angle64::from_radians(-half_theta), &[(control, target)]),
        Gate::rz(Angle64::from_radians(half_theta), &[target]),
    ]);
    gates
}

/// Lower `CRX(theta) = (I (x) H) . CRZ(theta) . (I (x) H)`.
#[must_use]
pub fn lower_crx(theta_radians: f64, control: QubitId, target: QubitId) -> SmallVec<[Gate; 5]> {
    let mut gates = smallvec![Gate::h(&[target])];
    gates.extend(lower_crz(theta_radians, control, target));
    gates.push(Gate::h(&[target]));
    gates
}

/// Lower `CRY(theta) = (I (x) SXdg) . CRZ(theta) . (I (x) SX)`.
///
/// Circuit emission order is `SX`, the `CRZ` lowering, then `SXdg`.
#[must_use]
pub fn lower_cry(theta_radians: f64, control: QubitId, target: QubitId) -> SmallVec<[Gate; 5]> {
    let mut gates = smallvec![Gate::sx(&[target])];
    gates.extend(lower_crz(theta_radians, control, target));
    gates.push(Gate::sxdg(&[target]));
    gates
}

/// Lower controlled phase while retaining its relative phase.
///
/// The identity is
/// `CPhase(lambda) = (U(0,0,lambda/2) (x) RZ(lambda/2)) . RZZ(-lambda/2)`.
/// `U(0,0,lambda/2)` carries the `exp(i lambda/4)` factor. `CPhase` is
/// 2π-periodic, so `lambda` is first reduced to its signed representative in
/// `(-π, π]`; every halved angle then lies in `(-π/2, π/2]`, away from the ±π
/// pair that a stored [`Angle64`] cannot tell apart, and the lowering is exact
/// for every input.
#[must_use]
pub fn lower_cphase(lambda_radians: f64, control: QubitId, target: QubitId) -> [Gate; 3] {
    let half_lambda = Angle64::from_radians(lambda_radians).to_radians_signed() / 2.0;
    let half_angle = Angle64::from_radians(half_lambda);
    [
        Gate::rzz(Angle64::from_radians(-half_lambda), &[(control, target)]),
        Gate::u(Angle64::ZERO, Angle64::ZERO, half_angle, &[control]),
        Gate::rz(half_angle, &[target]),
    ]
}

/// Lower a phase on zero, one, or two all-one operands to hardware gates.
///
/// A zero-operand phase is a scalar, so this returns no gates; the caller must
/// retain that scalar if global phase is observable in its representation. A
/// one-operand phase becomes exactly `U(0, 0, gamma)`. A two-operand phase uses
/// [`lower_cphase`] so its phase-carrying `U` leg is preserved.
///
/// # Errors
/// Returns an error if the phase exceeds the two-operand direct hardware lowering
/// limit (the operator itself is valid), or an operand is repeated.
pub fn lower_phase(gamma_radians: f64, qubits: &[QubitId]) -> Result<Vec<Gate>, PhaseGateError> {
    Ok(match qubits {
        [] => Vec::new(),
        &[qubit] => vec![Gate::u(
            Angle64::ZERO,
            Angle64::ZERO,
            Angle64::from_radians(gamma_radians),
            &[qubit],
        )],
        &[control, target] => {
            crate::unitary_rep::validate_phase_qubits(&[control.index(), target.index()])?;
            lower_cphase(gamma_radians, control, target).to_vec()
        }
        _ => {
            return Err(PhaseGateError::TooManyQubits {
                num_qubits: qubits.len(),
            });
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate_type::GateType;
    use num_complex::Complex64;

    #[test]
    fn lower_phase_rejects_unsupported_arity_without_panicking() {
        for num_qubits in [3, 200, 256] {
            let qubits: Vec<QubitId> = (0..num_qubits).map(QubitId).collect();
            assert_eq!(
                lower_phase(0.37, &qubits),
                Err(PhaseGateError::TooManyQubits { num_qubits })
            );
        }
    }

    #[test]
    fn lower_phase_rejects_duplicate_operands_without_panicking() {
        assert_eq!(
            lower_phase(0.37, &[QubitId(7), QubitId(7)]),
            Err(PhaseGateError::DuplicateQubit { qubit: 7 })
        );
    }

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
            GateType::Z => {
                for (basis, amplitude) in state.iter_mut().enumerate() {
                    if basis & (1 << (1 - gate.qubits[0].index())) != 0 {
                        *amplitude = -*amplitude;
                    }
                }
            }
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
    ) {
        let (phase_column, phase_row, reference) = (0..4)
            .flat_map(|column| (0..4).map(move |row| (column, row)))
            .map(|(column, row)| (column, row, expected[column][row]))
            .max_by(|left, right| left.2.norm().total_cmp(&right.2.norm()))
            .expect("matrix has entries");
        let phase = actual[phase_column][phase_row] / reference;
        assert!((phase.norm() - 1.0).abs() < TOLERANCE);
        assert!((phase - Complex64::new(1.0, 0.0)).norm() < TOLERANCE);
        for column in 0..4 {
            for row in 0..4 {
                assert!(
                    (actual[column][row] - expected[column][row]).norm() < TOLERANCE,
                    "column {column}, row {row}: actual={}, expected={}",
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
            -std::f64::consts::TAU,
            3.0 * std::f64::consts::TAU,
            -3.0 * std::f64::consts::TAU,
            3.0 * std::f64::consts::PI,
        ] {
            assert_matrix_eq_up_to_one_global_phase(
                matrix_from_lowering(&lower_crz(theta, QubitId(0), QubitId(1))),
                controlled_reference('Z', theta),
            );
            assert_matrix_eq_up_to_one_global_phase(
                matrix_from_lowering(&lower_crx(theta, QubitId(0), QubitId(1))),
                controlled_reference('X', theta),
            );
            assert_matrix_eq_up_to_one_global_phase(
                matrix_from_lowering(&lower_cry(theta, QubitId(0), QubitId(1))),
                controlled_reference('Y', theta),
            );
        }
    }

    fn axis_lowering(axis: char, theta: f64) -> SmallVec<[Gate; 5]> {
        match axis {
            'X' => lower_crx(theta, QubitId(0), QubitId(1)),
            'Y' => lower_cry(theta, QubitId(0), QubitId(1)),
            'Z' => lower_crz(theta, QubitId(0), QubitId(1))
                .into_iter()
                .collect(),
            _ => unreachable!(),
        }
    }

    fn matrix_distance(left: [[Complex64; 4]; 4], right: [[Complex64; 4]; 4]) -> f64 {
        left.iter()
            .flatten()
            .zip(right.iter().flatten())
            .map(|(l, r)| (l - r).norm_sqr())
            .sum::<f64>()
            .sqrt()
    }

    #[test]
    fn crz_two_pi_is_z_on_control() {
        let expected = matrix_from_lowering(&[Gate::z(&[QubitId(0)])]);
        assert_matrix_eq_up_to_one_global_phase(
            matrix_from_lowering(&lower_crz(TAU, QubitId(0), QubitId(1))),
            expected,
        );
    }

    #[test]
    fn controlled_rotations_are_continuous_at_odd_turns() {
        for axis in ['X', 'Y', 'Z'] {
            for theta in [TAU, -TAU, 3.0 * TAU, -3.0 * TAU] {
                let at = matrix_from_lowering(&axis_lowering(axis, theta));
                for offset in [-1e-7, 1e-7] {
                    let near = matrix_from_lowering(&axis_lowering(axis, theta + offset));
                    assert!(
                        matrix_distance(at, near) < 1e-7,
                        "axis={axis}, theta={theta}, offset={offset}"
                    );
                }
            }
        }
    }

    #[test]
    fn controlled_rotations_preserve_four_pi_periodicity() {
        for axis in ['X', 'Y', 'Z'] {
            for theta in [-200.0, -66.0, -3.0 * TAU, -PI, 0.37, TAU, 3.0 * TAU, 200.0] {
                assert_matrix_eq_up_to_one_global_phase(
                    matrix_from_lowering(&axis_lowering(axis, theta)),
                    matrix_from_lowering(&axis_lowering(axis, theta + 2.0 * TAU)),
                );
            }
        }
    }

    #[test]
    fn reduced_half_parity_matches_trigonometric_sheet_over_wide_ranges() {
        // Independent oracle: cos(theta/2) changes sign once per removed turn.
        // Midpoint sampling avoids the ambiguous zero at odd multiples of pi.
        for limit in [200.0, 10_000.0] {
            for sample in 0..300_000 {
                let theta = -limit + 2.0 * limit * (f64::from(sample) + 0.5) / 300_000.0;
                let (half, needs_z) = reduced_half_and_wrap_parity(theta);
                assert!(half > -PI / 2.0 && half <= PI / 2.0);
                assert_eq!(needs_z, (theta / 2.0).cos() < 0.0, "theta={theta}");
            }
        }
    }

    #[test]
    fn reduced_lowering_agrees_with_raw_half_away_from_odd_turns() {
        for sample in 0..800 {
            let theta = -200.0 + 400.0 * (f64::from(sample) + 0.5) / 800.0;
            let old = [
                Gate::rzz(
                    Angle64::from_radians(-theta / 2.0),
                    &[(QubitId(0), QubitId(1))],
                ),
                Gate::rz(Angle64::from_radians(theta / 2.0), &[QubitId(1)]),
            ];
            assert_matrix_eq_up_to_one_global_phase(
                matrix_from_lowering(&lower_crz(theta, QubitId(0), QubitId(1))),
                matrix_from_lowering(&old),
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
            );
        }
    }

    #[test]
    fn zero_qubit_phase_lowering_emits_no_hardware_gates() {
        assert!(lower_phase(0.37, &[]).unwrap().is_empty());
    }
}
