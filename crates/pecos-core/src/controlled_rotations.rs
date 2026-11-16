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

use crate::{Angle64, Gate, NonFiniteAngleError, PhaseGateError, QubitId};
use smallvec::{SmallVec, smallvec};
use std::f64::consts::{PI, TAU};

/// Halve the signed principal magnitude before quantizing it once, and return
/// the parity of removed turns separately. An Angle alone loses that bit.
fn reduced_half_and_wrap_parity(
    theta_radians: f64,
) -> Result<(Angle64, bool), NonFiniteAngleError> {
    // Validate the original input so errors retain its value and unit.
    Angle64::try_from_radians(theta_radians)?;
    // A signed remainder avoids adding then subtracting TAU for tiny negatives.
    // Keep +pi at the closed end of the principal interval (-pi, pi].
    let reduced = theta_radians % TAU;
    let signed = if reduced > PI {
        reduced - TAU
    } else if reduced <= -PI {
        reduced + TAU
    } else {
        reduced
    };
    // Quantizing a tiny negative via an unsigned floating-point remainder would
    // erase it. Halving in f64 is exact except for subnormal underflow; those
    // subnormal halves are far below one Angle64 tick and correctly round to zero.
    // Quantize the half once, then use integer negation for both signs and legs.
    let magnitude_half = Angle64::try_from_radians(signed.abs() / 2.0)?;
    let half = if signed < 0.0 {
        -magnitude_half
    } else {
        magnitude_half
    };
    // Use the 4pi reduction only for parity. Subtracting an even number of 2pi
    // turns preserves parity while keeping the rounded count in 0, 1, or 2.
    let bounded = theta_radians.rem_euclid(2.0 * TAU);
    let wraps = ((bounded - signed) / TAU).round();
    let needs_z = wraps.rem_euclid(2.0).total_cmp(&1.0).is_eq();
    Ok((half, needs_z))
}

/// Lower `CRZ(theta)` to native rotations.
///
/// The identity is
/// `CRZ(theta) = (I (x) RZ(theta/2)) . RZZ(-theta/2)`, with no global phase
/// before angle reduction. Reduce `theta` to `(-π, π]` before halving so neither
/// stored leg reaches the ambiguous ±π pair. Each removed 2π turn contributes
/// a `Z` on the control; retaining its parity preserves the exact global phase
/// and 4π periodicity. Halve the reduced magnitude before quantizing it once into
/// [`Angle64`], rounding to the nearest half-angle fraction (ties away from zero).
/// The two stored legs are exact fixed-point negatives. Half-angle quantization
/// contributes at most half a fraction unit, in addition to f64 argument-reduction
/// and radians-to-turns conversion error.
/// A separate reduction modulo 4π keeps the turn count bounded without changing
/// the signed representative. This prevents loss of integer
/// parity in an enormous quotient; it does not provide exact real-π argument
/// reduction for large floating-point inputs.
///
/// # Numerical validation
///
/// An 80-digit reference comparison of 10,000 random f64 inputs per magnitude
/// found no sheet errors through `|theta| <= 1e13`. The review measured about
/// 1 error per 10,000 at `1e14`; an independent sweep (seed 670) found 6. Errors
/// increase as floating-point input rounding and reduction lose angular detail.
/// These are sampled results, not a uniform bound: an evenly spaced sweep
/// already finds the wrong sheet at `theta = 156_700_000_000.0`. The regression
/// tests pin high-precision samples through `1e13` and the bounded outputs of
/// known counterexamples. No exact argument reduction is performed here.
/// # Errors
/// Rejects a non-finite angle, naming its value and unit.
pub fn lower_crz(
    theta_radians: f64,
    control: QubitId,
    target: QubitId,
) -> Result<SmallVec<[Gate; 3]>, NonFiniteAngleError> {
    let (half_theta, needs_z) = reduced_half_and_wrap_parity(theta_radians)?;
    let mut gates = SmallVec::new();
    if needs_z {
        gates.push(Gate::z(&[control]));
    }
    gates.extend([
        Gate::rzz(-half_theta, &[(control, target)]),
        Gate::rz(half_theta, &[target]),
    ]);
    Ok(gates)
}

/// Lower `CRX(theta) = (I (x) H) . CRZ(theta) . (I (x) H)`.
/// # Errors
/// Rejects a non-finite angle, naming its value and unit.
pub fn lower_crx(
    theta_radians: f64,
    control: QubitId,
    target: QubitId,
) -> Result<SmallVec<[Gate; 5]>, NonFiniteAngleError> {
    let mut gates = smallvec![Gate::h(&[target])];
    gates.extend(lower_crz(theta_radians, control, target)?);
    gates.push(Gate::h(&[target]));
    Ok(gates)
}

/// Lower `CRY(theta) = (I (x) SXdg) . CRZ(theta) . (I (x) SX)`.
///
/// Circuit emission order is `SX`, the `CRZ` lowering, then `SXdg`.
/// # Errors
/// Rejects a non-finite angle, naming its value and unit.
pub fn lower_cry(
    theta_radians: f64,
    control: QubitId,
    target: QubitId,
) -> Result<SmallVec<[Gate; 5]>, NonFiniteAngleError> {
    let mut gates = smallvec![Gate::sx(&[target])];
    gates.extend(lower_crz(theta_radians, control, target)?);
    gates.push(Gate::sxdg(&[target]));
    Ok(gates)
}

/// Lower controlled phase while retaining its relative phase.
///
/// The identity is
/// `CPhase(lambda) = (U(0,0,lambda/2) (x) RZ(lambda/2)) . RZZ(-lambda/2)`.
/// `U(0,0,lambda/2)` carries the `exp(i lambda/4)` factor. `CPhase` is
/// 2π-periodic, so `lambda` is first reduced to its signed representative in
/// `(-π, π]`; every halved angle then lies in `(-π/2, π/2]`, away from the ±π
/// pair that a stored [`Angle64`] cannot tell apart. The floating-point half is
/// quantized once to the nearest fraction unit, with ties away from zero.
/// # Errors
/// Rejects a non-finite angle, naming its value and unit.
pub fn lower_cphase(
    lambda_radians: f64,
    control: QubitId,
    target: QubitId,
) -> Result<[Gate; 3], NonFiniteAngleError> {
    let (half, _) = reduced_half_and_wrap_parity(lambda_radians)?;
    Ok(lower_cphase_half(half, control, target))
}

fn lower_cphase_angle(lambda: Angle64, control: QubitId, target: QubitId) -> [Gate; 3] {
    lower_cphase_half(lambda.signed_half(), control, target)
}

fn lower_cphase_half(half_angle: Angle64, control: QubitId, target: QubitId) -> [Gate; 3] {
    [
        Gate::rzz(-half_angle, &[(control, target)]),
        Gate::u(Angle64::ZERO, Angle64::ZERO, half_angle, &[control]),
        Gate::rz(half_angle, &[target]),
    ]
}

/// Lower a phase on the all-one subspace to hardware gates.
///
/// A zero-operand phase is a scalar and is rejected unless its normalized angle
/// is zero: a `Gate` list has no global-phase carrier. Once a carrier exists,
/// nontrivial scalars should be emitted instead. A
/// one-operand phase becomes exactly `U(0, 0, gamma)`. A two-operand phase uses
/// [`lower_cphase`] so its phase-carrying `U` leg is preserved. A three-operand
/// phase at exactly π lowers to one `CCZ`.
///
/// # Errors
/// Returns an error if the phase exceeds the two-operand direct hardware lowering
/// limit, except for the three-operand π case (the operator itself is valid),
/// an operand is repeated, or a nontrivial
/// zero-operand phase cannot be represented.
pub fn lower_phase(gamma_radians: f64, qubits: &[QubitId]) -> Result<Vec<Gate>, PhaseGateError> {
    // Preserve the source precision for the two-operand floating-point lowering.
    // Stored-angle callers below retain signed_half's fixed-point rounding.
    let gamma = Angle64::try_from_radians(gamma_radians)?;
    if let &[control, target] = qubits {
        crate::unitary_rep::validate_phase_qubits(&[control.index(), target.index()])?;
        return Ok(lower_cphase(gamma_radians, control, target)?.to_vec());
    }
    lower_phase_angle(gamma, qubits)
}

/// Lower a stored phase without converting through floating point.
/// Two-operand phases halve the signed principal fraction toward zero.
///
/// # Errors
/// Rejects duplicate operands, unsupported arities and nonidentity scalar phases,
/// with the same rules as [`lower_phase`].
pub fn lower_phase_angle(gamma: Angle64, qubits: &[QubitId]) -> Result<Vec<Gate>, PhaseGateError> {
    Ok(match qubits {
        [] if gamma != Angle64::ZERO => {
            return Err(PhaseGateError::UnrepresentableGlobalPhase { phase: gamma });
        }
        [] => Vec::new(),
        &[qubit] => vec![Gate::u(Angle64::ZERO, Angle64::ZERO, gamma, &[qubit])],
        &[control, target] => {
            crate::unitary_rep::validate_phase_qubits(&[control.index(), target.index()])?;
            lower_cphase_angle(gamma, control, target).to_vec()
        }
        &[a, b, c] if gamma == Angle64::HALF_TURN => {
            crate::unitary_rep::validate_phase_qubits(&[a.index(), b.index(), c.index()])?;
            vec![Gate::ccz(&[(a, b, c)])]
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

    #[test]
    fn lower_phase_three_operand_pi_is_ccz() {
        let qubits = [QubitId(4), QubitId(0), QubitId(2)];
        assert_eq!(lower_phase(PI, &qubits), Ok(vec![Gate::ccz(&[(4, 0, 2)])]));
        assert_eq!(
            lower_phase(PI / 2.0, &qubits),
            Err(PhaseGateError::TooManyQubits { num_qubits: 3 })
        );
        assert_eq!(
            lower_phase(PI, &[QubitId(4), QubitId(0), QubitId(4)]),
            Err(PhaseGateError::DuplicateQubit { qubit: 4 })
        );
        for num_qubits in [4, 200, 256] {
            let qubits: Vec<QubitId> = (0..num_qubits).map(QubitId).collect();
            assert_eq!(
                lower_phase(PI, &qubits),
                Err(PhaseGateError::TooManyQubits { num_qubits })
            );
        }
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
                matrix_from_lowering(&lower_crz(theta, QubitId(0), QubitId(1)).unwrap()),
                controlled_reference('Z', theta),
            );
            assert_matrix_eq_up_to_one_global_phase(
                matrix_from_lowering(&lower_crx(theta, QubitId(0), QubitId(1)).unwrap()),
                controlled_reference('X', theta),
            );
            assert_matrix_eq_up_to_one_global_phase(
                matrix_from_lowering(&lower_cry(theta, QubitId(0), QubitId(1)).unwrap()),
                controlled_reference('Y', theta),
            );
        }
    }

    fn axis_lowering(axis: char, theta: f64) -> SmallVec<[Gate; 5]> {
        match axis {
            'X' => lower_crx(theta, QubitId(0), QubitId(1)).unwrap(),
            'Y' => lower_cry(theta, QubitId(0), QubitId(1)).unwrap(),
            'Z' => lower_crz(theta, QubitId(0), QubitId(1))
                .unwrap()
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
            matrix_from_lowering(&lower_crz(TAU, QubitId(0), QubitId(1)).unwrap()),
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
                let (half, needs_z) = reduced_half_and_wrap_parity(theta).unwrap();
                assert!(
                    half.to_radians_signed() > -PI / 2.0 && half.to_radians_signed() <= PI / 2.0
                );
                assert_eq!(needs_z, (theta / 2.0).cos() < 0.0, "theta={theta}");
            }
        }
    }

    #[test]
    fn crz_just_above_negative_pi_keeps_two_gates() {
        let theta = (-PI).next_up();
        let (half, needs_z) = reduced_half_and_wrap_parity(theta).unwrap();
        assert_eq!(half, -Angle64::new(4_611_686_018_427_387_252));
        assert!(!needs_z);
        let gates = lower_crz(theta, QubitId(0), QubitId(1)).unwrap();
        assert_eq!(gates.len(), 2);
        assert_eq!(
            gates.as_slice(),
            &[
                Gate::rzz(
                    Angle64::try_from_radians(-theta / 2.0).unwrap(),
                    &[(QubitId(0), QubitId(1))]
                ),
                Gate::rz(-Angle64::new(4_611_686_018_427_387_252), &[QubitId(1)]),
            ]
        );
    }

    /// The principal representative is half-open at `+π`, so `π` itself keeps the
    /// positive branch and emits no correction. Choosing `-π` instead denotes the
    /// same operator, because the extra control `Z` cancels the sign it introduces,
    /// but it emits three gates where two suffice and so changes tick scheduling.
    #[test]
    fn crz_at_positive_pi_keeps_the_upper_branch_and_two_gates() {
        let (half, needs_z) = reduced_half_and_wrap_parity(PI).unwrap();
        assert_eq!(half, Angle64::QUARTER_TURN);
        assert!(!needs_z);

        let gates = lower_crz(PI, QubitId(0), QubitId(1)).unwrap();
        assert_eq!(gates.len(), 2);
        assert_eq!(
            gates.as_slice(),
            &[
                Gate::rzz(Angle64::THREE_QUARTERS_TURN, &[(QubitId(0), QubitId(1))]),
                Gate::rz(Angle64::QUARTER_TURN, &[QubitId(1)]),
            ]
        );

        // `-π` is the other end of the same half-open interval: it reduces onto the
        // upper branch and pays for it with the correction.
        let (negated_half, negated_needs_z) = reduced_half_and_wrap_parity(-PI).unwrap();
        assert_eq!(negated_half, Angle64::QUARTER_TURN);
        assert!(negated_needs_z);
        assert_eq!(lower_crz(-PI, QubitId(0), QubitId(1)).unwrap().len(), 3);
    }

    #[test]
    fn tiny_negative_crz_preserves_signed_residual_and_fixed_half() {
        let (half, needs_z) = reduced_half_and_wrap_parity(-1e-15).unwrap();
        assert_eq!(half, -Angle64::new(1468));
        assert!(!needs_z);
    }

    #[test]
    fn bounded_parity_matches_high_precision_samples_through_1e13() {
        // Reference: ceil((exact_f64(theta) - pi) / (2*pi)) modulo 2,
        // evaluated with 80-digit Decimal arithmetic. Factors are applied in f64
        // before conversion to Decimal. These samples establish no uniform bound.
        for (magnitude, expected) in [
            (1e0, [false, false, false, false]),
            (1e1, [false, true, true, false]),
            (1e2, [false, false, false, false]),
            (1e3, [false, true, true, true]),
            (1e4, [true, true, false, false]),
            (1e5, [true, true, true, true]),
            (1e6, [false, true, false, true]),
            (1e7, [false, true, false, true]),
            (1e8, [true, true, true, false]),
            (1e9, [false, true, true, true]),
            (1e10, [true, true, true, true]),
            (1e11, [true, false, false, true]),
            (1e12, [false, false, true, false]),
            (1e13, [true, false, true, true]),
        ] {
            for (factor, parity) in [0.125, 0.37, 0.75, 1.0].into_iter().zip(expected) {
                for sign in [-1.0, 1.0] {
                    let theta = sign * magnitude * factor;
                    let (half, needs_z) = reduced_half_and_wrap_parity(theta).unwrap();
                    assert_eq!(needs_z, parity, "theta={theta}");
                    assert!(
                        half.to_radians_signed() > -PI / 2.0
                            && half.to_radians_signed() <= PI / 2.0
                    );
                }
            }
        }
    }

    #[test]
    fn bounded_reduction_pins_large_inputs_and_a_sheet_counterexample() {
        // Pin outputs of the bounded f64 implementation, not ideal real-angle
        // answers. These two inputs have odd reference parity, but their reduced
        // f64 angles still lose accuracy. Fractions below are independently rounded
        // exact significand ratios of the reduced f64 half and f64 TAU.
        for (theta, rzz, rz) in [
            (1e20, 15_664_121_865_801_653_476, 2_782_622_207_907_898_140),
            (
                18_014_398_509_482_004.0,
                3_526_899_157_453_711_253,
                14_919_844_916_255_840_363,
            ),
        ] {
            let gates = lower_crz(theta, QubitId(0), QubitId(1)).unwrap();
            assert_eq!(gates.len(), 3);
            assert_eq!(gates[0], Gate::z(&[QubitId(0)]));
            assert_eq!(gates[1].angles[0], Angle64::new(rzz));
            assert_eq!(gates[2].angles[0], Angle64::new(rz));
        }
        // Even a value below 1e13 can lie close enough to a sheet boundary for
        // f64 reduction to choose the wrong parity. Do not promise a uniform bound.
        assert!(reduced_half_and_wrap_parity(156_700_000_000.0).unwrap().1);
    }

    #[test]
    fn reduced_lowering_agrees_with_raw_half_away_from_odd_turns() {
        for sample in 0..800 {
            let theta = -200.0 + 400.0 * (f64::from(sample) + 0.5) / 800.0;
            let old = [
                Gate::rzz(
                    Angle64::try_from_radians(-theta / 2.0).unwrap(),
                    &[(QubitId(0), QubitId(1))],
                ),
                Gate::rz(
                    Angle64::try_from_radians(theta / 2.0).unwrap(),
                    &[QubitId(1)],
                ),
            ];
            assert_matrix_eq_up_to_one_global_phase(
                matrix_from_lowering(&lower_crz(theta, QubitId(0), QubitId(1)).unwrap()),
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
                matrix_from_lowering(&lower_cphase(lambda, QubitId(0), QubitId(1)).unwrap()),
                expected,
            );
        }
    }

    #[test]
    fn zero_qubit_phase_lowering_rejects_nontrivial_scalars() {
        for radians in [0.37, -0.37, PI, -PI, PI / 2.0] {
            assert_eq!(
                lower_phase(radians, &[]),
                Err(PhaseGateError::UnrepresentableGlobalPhase {
                    phase: Angle64::try_from_radians(radians).unwrap(),
                })
            );
        }
        for radians in [0.0, -0.0, TAU, -TAU] {
            assert_eq!(lower_phase(radians, &[]), Ok(vec![]));
        }
    }
}
