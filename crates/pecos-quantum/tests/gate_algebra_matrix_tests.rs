// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Matrix-level verification of cross-type gate algebra operators.
//!
//! Each test builds a result via the gate algebra operators (`*`, `&`) on
//! base types (Pauli, Clifford, Unitary) and verifies it is matrix-equivalent
//! to the independently constructed reference built from `UnitaryRep` constructors.

use pecos_core::clifford::Clifford;
use pecos_core::unitary_rep::{self, RotationType, Unitary, UnitaryRep};
use pecos_core::{Angle64, Pauli, PauliString};
use pecos_quantum::unitary_matrix::unitaries_equiv;

// ============================================================================
// Pauli * Pauli: verify via PauliString -> UnitaryRep matrix
// ============================================================================

#[test]
fn matrix_pauli_mul_xx_is_identity() {
    let result: PauliString = Pauli::X * Pauli::X;
    let result_ur = UnitaryRep::from(result);
    let reference = unitary_rep::I(0);
    assert!(
        unitaries_equiv(&result_ur, &reference),
        "X * X should equal I"
    );
}

#[test]
fn matrix_pauli_mul_xz_is_neg_iy() {
    let result: PauliString = Pauli::X * Pauli::Z;
    let result_ur = UnitaryRep::from(result);
    // X * Z = -iY
    let reference = -pecos_core::unitary_rep::i * unitary_rep::Y(0);
    assert!(
        unitaries_equiv(&result_ur, &reference),
        "X * Z should equal -iY"
    );
}

#[test]
fn matrix_pauli_mul_yz_is_ix() {
    let result: PauliString = Pauli::Y * Pauli::Z;
    let result_ur = UnitaryRep::from(result);
    // Y * Z = iX
    let reference = pecos_core::unitary_rep::i * unitary_rep::X(0);
    assert!(
        unitaries_equiv(&result_ur, &reference),
        "Y * Z should equal iX"
    );
}

#[test]
fn matrix_pauli_mul_zx_is_iy() {
    let result: PauliString = Pauli::Z * Pauli::X;
    let result_ur = UnitaryRep::from(result);
    // Z * X = iY
    let reference = pecos_core::unitary_rep::i * unitary_rep::Y(0);
    assert!(
        unitaries_equiv(&result_ur, &reference),
        "Z * X should equal iY"
    );
}

// ============================================================================
// Pauli & Pauli: tensor product matrix verification
// ============================================================================

#[test]
fn matrix_pauli_tensor_xz() {
    let result: PauliString = Pauli::X & Pauli::Z;
    let result_ur = UnitaryRep::from(result);
    let reference = unitary_rep::X(0) & unitary_rep::Z(1);
    assert!(
        unitaries_equiv(&result_ur, &reference),
        "X & Z should equal X(0) tensor Z(1)"
    );
}

#[test]
fn matrix_pauli_tensor_yz() {
    let result: PauliString = Pauli::Y & Pauli::Z;
    let result_ur = UnitaryRep::from(result);
    let reference = unitary_rep::Y(0) & unitary_rep::Z(1);
    assert!(
        unitaries_equiv(&result_ur, &reference),
        "Y & Z should equal Y(0) tensor Z(1)"
    );
}

// ============================================================================
// Pauli * Clifford -> CliffordRep: verify via UnitaryRep matrix
// ============================================================================

/// Helper: verify a `CliffordRep` matches a reference `UnitaryRep` by checking
/// that both produce the same `CliffordRep` (stabilizer images).
fn assert_clifford_rep_matches_unitary(
    cr: &pecos_core::clifford_rep::CliffordRep,
    reference: &UnitaryRep,
    nq: usize,
    label: &str,
) {
    let ref_cr = reference
        .to_clifford_rep(nq)
        .unwrap_or_else(|| panic!("{label}: reference UnitaryRep should be Clifford"));
    assert_eq!(*cr, ref_cr, "{label}: CliffordRep does not match reference");
}

#[test]
fn matrix_pauli_x_mul_clifford_h() {
    // X * H: apply H first, then X
    let result = Pauli::X * Clifford::H;
    let reference = unitary_rep::X(0) * unitary_rep::H(0);
    assert_clifford_rep_matches_unitary(&result, &reference, 1, "X * H");
}

#[test]
fn matrix_clifford_h_mul_pauli_x() {
    // H * X: apply X first, then H
    let result = Clifford::H * Pauli::X;
    let reference = unitary_rep::H(0) * unitary_rep::X(0);
    assert_clifford_rep_matches_unitary(&result, &reference, 1, "H * X");
}

#[test]
fn matrix_composition_order_xh_vs_hx() {
    // Verify X*H and H*X produce different matrices
    let xh = unitary_rep::X(0) * unitary_rep::H(0);
    let hx = unitary_rep::H(0) * unitary_rep::X(0);
    assert!(
        !unitaries_equiv(&xh, &hx),
        "X*H and H*X should NOT be equivalent"
    );
}

#[test]
fn matrix_pauli_z_mul_clifford_sz() {
    let result = Pauli::Z * Clifford::SZ;
    let reference = unitary_rep::Z(0) * unitary_rep::SZ(0);
    assert_clifford_rep_matches_unitary(&result, &reference, 1, "Z * SZ");
}

#[test]
fn matrix_identity_mul_clifford_h() {
    let result = Pauli::I * Clifford::H;
    let reference = unitary_rep::H(0);
    assert_clifford_rep_matches_unitary(&result, &reference, 1, "I * H");
}

// ============================================================================
// Pauli & Clifford -> CliffordRep: tensor product matrix verification
// ============================================================================

#[test]
fn matrix_pauli_x_tensor_clifford_h() {
    let result = Pauli::X & Clifford::H;
    let reference = unitary_rep::X(0) & unitary_rep::H(1);
    assert_clifford_rep_matches_unitary(&result, &reference, 2, "X & H");
}

#[test]
fn matrix_clifford_cx_tensor_pauli_z() {
    let result = Clifford::CX & Pauli::Z;
    let reference = unitary_rep::CX(0, 1) & unitary_rep::Z(2);
    assert_clifford_rep_matches_unitary(&result, &reference, 3, "CX & Z");
}

#[test]
fn matrix_clifford_h_tensor_pauli_x() {
    let result = Clifford::H & Pauli::X;
    let reference = unitary_rep::H(0) & unitary_rep::X(1);
    assert_clifford_rep_matches_unitary(&result, &reference, 2, "H & X");
}

// ============================================================================
// Clifford & Clifford -> CliffordRep: tensor product matrix verification
// ============================================================================

#[test]
fn matrix_clifford_h_tensor_sz() {
    let result = Clifford::H & Clifford::SZ;
    let reference = unitary_rep::H(0) & unitary_rep::SZ(1);
    assert_clifford_rep_matches_unitary(&result, &reference, 2, "H & SZ");
}

#[test]
fn matrix_clifford_cx_tensor_h() {
    let result = Clifford::CX & Clifford::H;
    let reference = unitary_rep::CX(0, 1) & unitary_rep::H(2);
    assert_clifford_rep_matches_unitary(&result, &reference, 3, "CX & H");
}

#[test]
fn matrix_clifford_cx_tensor_cz() {
    let result = Clifford::CX & Clifford::CZ;
    let reference = unitary_rep::CX(0, 1) & unitary_rep::CZ(2, 3);
    assert_clifford_rep_matches_unitary(&result, &reference, 4, "CX & CZ");
}

// ============================================================================
// Unitary cross-type ops: matrix-level verification via unitaries_equiv
// ============================================================================

#[test]
fn matrix_pauli_x_mul_unitary_rz() {
    let rz = Unitary::Rotation {
        rotation_type: RotationType::RZ,
        angle: Angle64::from_turn_ratio(1, 8),
    };
    let result = Pauli::X * rz;
    let reference = unitary_rep::X(0) * unitary_rep::RZ(Angle64::from_turn_ratio(1, 8), 0);
    assert!(
        unitaries_equiv(&result, &reference),
        "Pauli::X * RZ(1/8) should match X(0) * RZ(1/8, 0)"
    );
}

#[test]
fn matrix_unitary_rz_mul_pauli_x() {
    let rz = Unitary::Rotation {
        rotation_type: RotationType::RZ,
        angle: Angle64::from_turn_ratio(1, 8),
    };
    let result = rz * Pauli::X;
    let reference = unitary_rep::RZ(Angle64::from_turn_ratio(1, 8), 0) * unitary_rep::X(0);
    assert!(
        unitaries_equiv(&result, &reference),
        "RZ(1/8) * Pauli::X should match RZ(1/8, 0) * X(0)"
    );
}

#[test]
fn matrix_pauli_x_tensor_unitary_rz() {
    let rz = Unitary::Rotation {
        rotation_type: RotationType::RZ,
        angle: Angle64::QUARTER_TURN,
    };
    let result = Pauli::X & rz;
    let reference = unitary_rep::X(0) & unitary_rep::RZ(Angle64::QUARTER_TURN, 1);
    assert!(
        unitaries_equiv(&result, &reference),
        "X & RZ should match X(0) tensor RZ(1)"
    );
}

#[test]
fn matrix_clifford_h_mul_unitary_rz() {
    let rz = Unitary::Rotation {
        rotation_type: RotationType::RZ,
        angle: Angle64::from_turn_ratio(1, 8),
    };
    let result = Clifford::H * rz;
    let reference = unitary_rep::H(0) * unitary_rep::RZ(Angle64::from_turn_ratio(1, 8), 0);
    assert!(
        unitaries_equiv(&result, &reference),
        "Clifford::H * RZ(1/8) should match H(0) * RZ(1/8, 0)"
    );
}

#[test]
fn matrix_unitary_rz_mul_clifford_h() {
    let rz = Unitary::Rotation {
        rotation_type: RotationType::RZ,
        angle: Angle64::from_turn_ratio(1, 8),
    };
    let result = rz * Clifford::H;
    let reference = unitary_rep::RZ(Angle64::from_turn_ratio(1, 8), 0) * unitary_rep::H(0);
    assert!(
        unitaries_equiv(&result, &reference),
        "RZ(1/8) * Clifford::H should match RZ(1/8, 0) * H(0)"
    );
}

#[test]
fn matrix_clifford_h_tensor_unitary_rz() {
    let rz = Unitary::Rotation {
        rotation_type: RotationType::RZ,
        angle: Angle64::QUARTER_TURN,
    };
    let result = Clifford::H & rz;
    let reference = unitary_rep::H(0) & unitary_rep::RZ(Angle64::QUARTER_TURN, 1);
    assert!(
        unitaries_equiv(&result, &reference),
        "H & RZ should match H(0) tensor RZ(1)"
    );
}

#[test]
fn matrix_unitary_rz_tensor_clifford_cx() {
    let rz = Unitary::Rotation {
        rotation_type: RotationType::RZ,
        angle: Angle64::QUARTER_TURN,
    };
    let result = rz & Clifford::CX;
    let reference = unitary_rep::RZ(Angle64::QUARTER_TURN, 0) & unitary_rep::CX(1, 2);
    assert!(
        unitaries_equiv(&result, &reference),
        "RZ & CX should match RZ(0) tensor CX(1,2)"
    );
}

// ============================================================================
// Unitary * Unitary and Unitary & Unitary matrix verification
// ============================================================================

#[test]
fn matrix_unitary_h_mul_rz() {
    let h = Unitary::Named(pecos_core::gate_type::GateType::H);
    let rz = Unitary::Rotation {
        rotation_type: RotationType::RZ,
        angle: Angle64::from_turn_ratio(1, 8),
    };
    let result = h * rz;
    let reference = unitary_rep::H(0) * unitary_rep::RZ(Angle64::from_turn_ratio(1, 8), 0);
    assert!(
        unitaries_equiv(&result, &reference),
        "Unitary H * Unitary RZ should match H(0) * RZ(0)"
    );
}

#[test]
fn matrix_unitary_h_tensor_cx() {
    let h = Unitary::Named(pecos_core::gate_type::GateType::H);
    let cx = Unitary::Named(pecos_core::gate_type::GateType::CX);
    let result = h & cx;
    let reference = unitary_rep::H(0) & unitary_rep::CX(1, 2);
    assert!(
        unitaries_equiv(&result, &reference),
        "Unitary H & CX should match H(0) tensor CX(1,2)"
    );
}

#[test]
fn matrix_unitary_cx_tensor_rz() {
    let cx = Unitary::Named(pecos_core::gate_type::GateType::CX);
    let rz = Unitary::Rotation {
        rotation_type: RotationType::RZ,
        angle: Angle64::QUARTER_TURN,
    };
    let result = cx & rz;
    let reference = unitary_rep::CX(0, 1) & unitary_rep::RZ(Angle64::QUARTER_TURN, 2);
    assert!(
        unitaries_equiv(&result, &reference),
        "CX & RZ should match CX(0,1) tensor RZ(2)"
    );
}
