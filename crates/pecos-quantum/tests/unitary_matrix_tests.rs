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

//! Tests for [`UnitaryMatrix`] operators and [`ToMatrix`] impls on base types.

use num_complex::Complex64;
use pecos_core::Phase;
use pecos_core::clifford::Clifford;
use pecos_core::clifford_rep::CliffordRep;
use pecos_core::unitary_rep::{RotationType, Unitary};
use pecos_core::{Angle64, Pauli, PauliString, op};
use pecos_quantum::unitary_matrix::{ToMatrix, UnitaryMatrix};

// --- UnitaryMatrix: construction and num_qubits ---

#[test]
fn identity_num_qubits() {
    assert_eq!(UnitaryMatrix::identity(2).num_qubits(), 1);
    assert_eq!(UnitaryMatrix::identity(4).num_qubits(), 2);
    assert_eq!(UnitaryMatrix::identity(8).num_qubits(), 3);
}

#[test]
fn from_dmatrix_roundtrip() {
    let dm = nalgebra::DMatrix::<Complex64>::identity(4, 4);
    let um = UnitaryMatrix::from(dm.clone());
    let back: nalgebra::DMatrix<Complex64> = um.into_inner();
    assert_eq!(dm, back);
}

// --- UnitaryMatrix: * (composition) ---

#[test]
fn matrix_mul_matches_unitary_rep_compose() {
    let h = pecos_core::unitary_rep::H(0);
    let x = pecos_core::unitary_rep::X(0);

    let via_rep = (h.clone() * x.clone()).to_matrix();
    let via_mat = h.to_matrix() * x.to_matrix();

    assert!(via_rep.equiv_up_to_phase(&via_mat));
}

#[test]
fn matrix_mul_ref_variants() {
    let a = Pauli::X.to_matrix();
    let b = Pauli::Z.to_matrix();

    let owned_owned = a.clone() * b.clone();
    let owned_ref = a.clone() * &b;
    let ref_owned = &a * b.clone();
    let ref_ref = &a * &b;

    assert!(owned_owned.equiv_up_to_phase(&owned_ref));
    assert!(owned_owned.equiv_up_to_phase(&ref_owned));
    assert!(owned_owned.equiv_up_to_phase(&ref_ref));
}

#[test]
fn matrix_mul_not_commutative() {
    let h = Clifford::H.to_matrix();
    let sx = Clifford::SX.to_matrix();

    let hsx = &h * &sx;
    let sxh = &sx * &h;

    assert!(!hsx.equiv_up_to_phase(&sxh));
}

// --- UnitaryMatrix: & (tensor / Kronecker) ---

#[test]
fn matrix_tensor_produces_correct_dimension() {
    let h = pecos_core::unitary_rep::H(0).to_matrix();
    let z = pecos_core::unitary_rep::Z(0).to_matrix();

    let t = &h & &z;
    // 2x2 kron 2x2 = 4x4
    assert_eq!(t.nrows(), 4);
    assert_eq!(t.num_qubits(), 2);
}

#[test]
fn matrix_tensor_is_kronecker_product() {
    // Verify UnitaryMatrix & is the standard Kronecker product:
    // (A kron B)|ij> = A|i> tensor B|j>
    let x = Pauli::X.to_matrix();
    let z = Pauli::Z.to_matrix();

    let xz = &x & &z;
    // kron(X, Z) has dimension 4x4
    // X = [[0,1],[1,0]], Z = [[1,0],[0,-1]]
    // kron(X, Z) = [[0,0,1,0],[0,0,0,-1],[1,0,0,0],[0,-1,0,0]]
    let one = Complex64::new(1.0, 0.0);
    let neg = Complex64::new(-1.0, 0.0);
    assert!((xz[(0, 2)] - one).norm() < 1e-10);
    assert!((xz[(1, 3)] - neg).norm() < 1e-10);
    assert!((xz[(2, 0)] - one).norm() < 1e-10);
    assert!((xz[(3, 1)] - neg).norm() < 1e-10);
}

#[test]
fn matrix_tensor_dimension() {
    let a = Pauli::X.to_matrix(); // 2x2
    let b = Pauli::Z.to_matrix(); // 2x2

    let t = &a & &b;
    assert_eq!(t.nrows(), 4);
    assert_eq!(t.num_qubits(), 2);
}

#[test]
fn matrix_tensor_ref_variants() {
    let a = Pauli::X.to_matrix();
    let b = Pauli::Y.to_matrix();

    let owned_owned = a.clone() & b.clone();
    let owned_ref = a.clone() & &b;
    let ref_owned = &a & b.clone();
    let ref_ref = &a & &b;

    assert!(owned_owned.equiv_up_to_phase(&owned_ref));
    assert!(owned_owned.equiv_up_to_phase(&ref_owned));
    assert!(owned_owned.equiv_up_to_phase(&ref_ref));
}

#[test]
fn matrix_tensor_not_commutative() {
    let x = Pauli::X.to_matrix();
    let z = Pauli::Z.to_matrix();

    let xz = &x & &z;
    let zx = &z & &x;

    assert!(!xz.equiv_up_to_phase(&zx));
}

// --- UnitaryMatrix: scalar multiplication ---

#[test]
fn matrix_scalar_complex64() {
    let x = Pauli::X.to_matrix();
    let phase = Complex64::new(0.0, 1.0); // i

    let right = &x * phase;
    let left = phase * &x;

    // i*X and X*i should be the same
    assert!(right.equiv_up_to_phase(&left));
    // i*X is not the same as X (different global phase, but equiv_up_to_phase allows that)
    assert!(right.equiv_up_to_phase(&x));
}

#[test]
fn matrix_scalar_f64() {
    let x = Pauli::X.to_matrix();

    let right = &x * 2.0;
    let left = 2.0 * &x;

    // 2*X from left and right should match
    assert!(right.equiv_up_to_phase(&left));
}

// --- UnitaryMatrix: subtraction and negation ---

#[test]
fn matrix_sub_self_is_zero() {
    let h = Clifford::H.to_matrix();
    let diff = &h - &h;
    assert!(diff.norm() < 1e-10);
}

#[test]
fn matrix_neg_double_is_identity_op() {
    let h = Clifford::H.to_matrix();
    let neg_neg = -(-&h);
    assert!(h.equiv_up_to_phase(&neg_neg));
}

#[test]
fn matrix_neg_ref_variant() {
    let x = Pauli::X.to_matrix();
    let neg_owned = -(x.clone());
    let neg_ref = -&x;
    assert!(neg_owned.equiv_up_to_phase(&neg_ref));
}

// --- UnitaryMatrix: adjoint ---

#[test]
fn matrix_adjoint_of_unitary_is_inverse() {
    let h = Clifford::H.to_matrix();
    let product = h.adjoint() * &h;
    let identity = UnitaryMatrix::identity(2);
    assert!((product - identity).norm() < 1e-10);
}

#[test]
fn matrix_adjoint_of_adjoint_is_original() {
    let sx = Clifford::SX.to_matrix();
    let double_adj = sx.adjoint().adjoint();
    assert!(sx.equiv_up_to_phase(&double_adj));
}

// --- UnitaryMatrix: equiv_up_to_phase ---

#[test]
fn equiv_up_to_phase_same_matrix() {
    let h = Clifford::H.to_matrix();
    assert!(h.equiv_up_to_phase(&h));
}

#[test]
fn equiv_up_to_phase_with_global_phase() {
    let x = Pauli::X.to_matrix();
    let ix = &x * Complex64::new(0.0, 1.0);
    assert!(x.equiv_up_to_phase(&ix));
}

#[test]
fn equiv_up_to_phase_different_gates() {
    let h = Clifford::H.to_matrix();
    let x = Pauli::X.to_matrix();
    assert!(!h.equiv_up_to_phase(&x));
}

// --- ToMatrix: Pauli base type ---

#[test]
fn to_matrix_pauli_x() {
    let mat = Pauli::X.to_matrix();
    assert_eq!(mat.num_qubits(), 1);
    // X = [[0, 1], [1, 0]]
    assert!((mat[(0, 0)]).norm() < 1e-10);
    assert!((mat[(0, 1)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
    assert!((mat[(1, 0)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
    assert!((mat[(1, 1)]).norm() < 1e-10);
}

#[test]
fn to_matrix_pauli_matches_unitary_rep() {
    for pauli in [Pauli::I, Pauli::X, Pauli::Y, Pauli::Z] {
        let base_mat = pauli.to_matrix();
        let rep_mat = pauli.on_qubit(0).to_matrix();
        assert!(
            base_mat.equiv_up_to_phase(&rep_mat),
            "Pauli::{pauli:?}.to_matrix() should match on_qubit(0).to_matrix()"
        );
    }
}

// --- ToMatrix: Clifford base type ---

#[test]
fn to_matrix_clifford_h() {
    let mat = Clifford::H.to_matrix();
    assert_eq!(mat.num_qubits(), 1);
    let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
    assert!((mat[(0, 0)] - Complex64::new(sqrt2_inv, 0.0)).norm() < 1e-10);
}

#[test]
fn to_matrix_clifford_all_1q_matches_unitary_rep() {
    for &cliff in Clifford::all_1q() {
        let base_mat = cliff.to_matrix();
        let rep_mat = cliff.on_qubit(0).to_matrix();
        assert!(
            base_mat.equiv_up_to_phase(&rep_mat),
            "Clifford::{cliff:?}.to_matrix() should match on_qubit(0).to_matrix()"
        );
    }
}

#[test]
fn to_matrix_clifford_2q() {
    let mat = Clifford::CX.to_matrix();
    assert_eq!(mat.num_qubits(), 2);

    let rep_mat = Clifford::CX.on_qubits(0, 1).to_matrix();
    assert!(mat.equiv_up_to_phase(&rep_mat));
}

#[test]
fn to_matrix_clifford_all_2q_gates() {
    for &cliff in Clifford::all_2q() {
        let base_mat = cliff.to_matrix();
        let rep_mat = cliff.on_qubits(0, 1).to_matrix();
        assert!(
            base_mat.equiv_up_to_phase(&rep_mat),
            "Clifford::{cliff:?}.to_matrix() should match on_qubits(0,1).to_matrix()"
        );
    }
}

// --- ToMatrix: Unitary base type ---

#[test]
fn to_matrix_unitary_named_1q() {
    let u = Unitary::Named(pecos_core::gate_type::GateType::H);
    let mat = u.to_matrix();
    assert_eq!(mat.num_qubits(), 1);

    let rep_mat = u.on_qubit(0).to_matrix();
    assert!(mat.equiv_up_to_phase(&rep_mat));
}

#[test]
fn to_matrix_unitary_named_all_1q_are_unitary() {
    use pecos_core::gate_type::GateType;
    let identity = UnitaryMatrix::identity(2);
    let gates_1q = [
        GateType::I,
        GateType::X,
        GateType::Y,
        GateType::Z,
        GateType::H,
        GateType::SX,
        GateType::SXdg,
        GateType::SY,
        GateType::SYdg,
        GateType::SZ,
        GateType::SZdg,
        GateType::F,
        GateType::Fdg,
        GateType::T,
        GateType::Tdg,
    ];
    for gt in gates_1q {
        let u = Unitary::Named(gt);
        let mat = u.to_matrix();
        assert_eq!(mat.num_qubits(), 1, "{gt:?} should be 1-qubit");
        let product = mat.adjoint() * &mat;
        let diff = (product - identity.clone()).norm();
        assert!(diff < 1e-10, "Named({gt:?}) is not unitary, diff = {diff}");
    }
}

#[test]
fn to_matrix_unitary_named_2q() {
    let u = Unitary::Named(pecos_core::gate_type::GateType::CX);
    let mat = u.to_matrix();
    assert_eq!(mat.num_qubits(), 2);

    let rep_mat = u.on_qubits(0, 1).to_matrix();
    assert!(mat.equiv_up_to_phase(&rep_mat));
}

#[test]
fn to_matrix_unitary_named_all_2q_are_unitary() {
    use pecos_core::gate_type::GateType;
    let identity = UnitaryMatrix::identity(4);
    let gates_2q = [
        GateType::CX,
        GateType::CY,
        GateType::CZ,
        GateType::CH,
        GateType::SWAP,
        GateType::SXX,
        GateType::SXXdg,
        GateType::SYY,
        GateType::SYYdg,
        GateType::SZZ,
        GateType::SZZdg,
    ];
    for gt in gates_2q {
        let u = Unitary::Named(gt);
        let mat = u.to_matrix();
        assert_eq!(mat.num_qubits(), 2, "{gt:?} should be 2-qubit");
        let product = mat.adjoint() * &mat;
        let diff = (product - identity.clone()).norm();
        assert!(diff < 1e-10, "Named({gt:?}) is not unitary, diff = {diff}");
    }
}

#[test]
fn to_matrix_unitary_all_1q_rotations() {
    let identity_2 = UnitaryMatrix::identity(2);
    for rot in [RotationType::RX, RotationType::RY, RotationType::RZ] {
        let u = Unitary::Rotation {
            rotation_type: rot,
            angle: Angle64::QUARTER_TURN,
        };
        let mat = u.to_matrix();
        assert_eq!(mat.num_qubits(), 1, "{rot:?} should be 1-qubit");

        // Verify unitarity
        let product = mat.adjoint() * &mat;
        let diff = (product - identity_2.clone()).norm();
        assert!(diff < 1e-10, "{rot:?} matrix is not unitary, diff = {diff}");

        // Verify matches on_qubit embedding
        let rep_mat = u.on_qubit(0).to_matrix();
        assert!(
            mat.equiv_up_to_phase(&rep_mat),
            "{rot:?}.to_matrix() should match on_qubit(0).to_matrix()"
        );
    }
}

#[test]
fn to_matrix_unitary_all_2q_rotations() {
    let identity_4 = UnitaryMatrix::identity(4);
    for rot in [RotationType::RXX, RotationType::RYY, RotationType::RZZ] {
        let u = Unitary::Rotation {
            rotation_type: rot,
            angle: Angle64::QUARTER_TURN,
        };
        let mat = u.to_matrix();
        assert_eq!(mat.num_qubits(), 2, "{rot:?} should be 2-qubit");

        // Verify unitarity
        let product = mat.adjoint() * &mat;
        let diff = (product - identity_4.clone()).norm();
        assert!(diff < 1e-10, "{rot:?} matrix is not unitary, diff = {diff}");

        // Verify matches on_qubits embedding
        let rep_mat = u.on_qubits(0, 1).to_matrix();
        assert!(
            mat.equiv_up_to_phase(&rep_mat),
            "{rot:?}.to_matrix() should match on_qubits(0,1).to_matrix()"
        );
    }
}

#[test]
fn to_matrix_unitary_rotation_at_various_angles() {
    let identity_2 = UnitaryMatrix::identity(2);
    let angles = [
        Angle64::ZERO,
        Angle64::QUARTER_TURN,
        Angle64::HALF_TURN,
        Angle64::THREE_QUARTERS_TURN,
    ];
    for angle in angles {
        let u = Unitary::Rotation {
            rotation_type: RotationType::RZ,
            angle,
        };
        let mat = u.to_matrix();
        let product = mat.adjoint() * &mat;
        let diff = (product - identity_2.clone()).norm();
        assert!(diff < 1e-10, "RZ({angle:?}) is not unitary, diff = {diff}");
    }
}

#[test]
fn to_matrix_unitary_named_ccx_is_unitary() {
    use pecos_core::gate_type::GateType;
    let u = Unitary::Named(GateType::CCX);
    let mat = u.to_matrix();
    assert_eq!(mat.num_qubits(), 3);
    let identity = UnitaryMatrix::identity(8);
    let product = mat.adjoint() * &mat;
    let diff = (product - identity).norm();
    assert!(diff < 1e-10, "CCX is not unitary, diff = {diff}");
}

#[test]
fn to_matrix_unitary_named_ccx_is_involution() {
    use pecos_core::gate_type::GateType;
    let mat = Unitary::Named(GateType::CCX).to_matrix();
    let product = &mat * &mat;
    let identity = UnitaryMatrix::identity(8);
    let diff = (product - identity).norm();
    assert!(diff < 1e-10, "CCX^2 should be identity, diff = {diff}");
}

#[test]
#[should_panic(expected = "requires angle parameter")]
fn to_matrix_unitary_named_rx_panics() {
    use pecos_core::gate_type::GateType;
    let _ = Unitary::Named(GateType::RX).to_matrix();
}

#[test]
#[should_panic(expected = "is not a unitary gate")]
fn to_matrix_unitary_named_mz_panics() {
    use pecos_core::gate_type::GateType;
    let _ = Unitary::Named(GateType::MZ).to_matrix();
}

// --- ToMatrix: CliffordRep ---

#[test]
fn to_matrix_clifford_rep_h() {
    let cr = Clifford::H.on_qubit(0);
    let cr_mat = cr.to_matrix();
    let ur_mat = pecos_core::unitary_rep::H(0).to_matrix();
    assert!(cr_mat.equiv_up_to_phase(&ur_mat));
}

#[test]
fn to_matrix_clifford_rep_cx() {
    let cr = Clifford::CX.on_qubits(0, 1);
    let cr_mat = cr.to_matrix();
    let ur_mat = pecos_core::unitary_rep::CX(0, 1).to_matrix();
    assert!(cr_mat.equiv_up_to_phase(&ur_mat));
}

#[test]
fn to_matrix_clifford_rep_tensor_product() {
    let cr = Clifford::H & Clifford::SZ;
    let cr_mat = cr.to_matrix();
    let ur_mat = (pecos_core::unitary_rep::H(0) & pecos_core::unitary_rep::SZ(1)).to_matrix();
    assert!(cr_mat.equiv_up_to_phase(&ur_mat));
}

// --- ToMatrix: Op ---

#[test]
fn to_matrix_op_clifford() {
    let mat = op::H(0).to_matrix();
    let expected = pecos_core::unitary_rep::H(0).to_matrix();
    assert!(mat.equiv_up_to_phase(&expected));
}

#[test]
fn to_matrix_op_2q() {
    let mat = op::CX(0, 1).to_matrix();
    let expected = pecos_core::unitary_rep::CX(0, 1).to_matrix();
    assert!(mat.equiv_up_to_phase(&expected));
}

#[test]
fn to_matrix_op_rotation() {
    let mat = op::RZ(Angle64::QUARTER_TURN, 0).to_matrix();
    let expected = pecos_core::unitary_rep::RZ(Angle64::QUARTER_TURN, 0).to_matrix();
    assert!(mat.equiv_up_to_phase(&expected));
}

#[test]
#[should_panic(expected = "Cannot convert non-unitary")]
fn to_matrix_op_channel_panics() {
    let channel = op::Depolarizing(0.1, 0);
    let _ = channel.to_matrix();
}

// --- Cross-level: matrix ops match algebraic ops ---

#[test]
fn matrix_compose_matches_algebra() {
    // H * X at the matrix level should match H * X at the UnitaryRep level
    let h_mat = Clifford::H.to_matrix();
    let x_mat = Pauli::X.to_matrix();

    let via_matrix = &h_mat * &x_mat;
    let via_algebra = (pecos_core::unitary_rep::H(0) * pecos_core::unitary_rep::X(0)).to_matrix();

    assert!(via_matrix.equiv_up_to_phase(&via_algebra));
}

#[test]
fn matrix_tensor_associative() {
    // (A kron B) kron C == A kron (B kron C)
    let x = Pauli::X.to_matrix();
    let y = Pauli::Y.to_matrix();
    let z = Pauli::Z.to_matrix();

    let left = (&x & &y) & &z;
    let right = &x & (&y & &z);

    assert!(left.equiv_up_to_phase(&right));
}

// --- All Cliffords: unitarity and dagger pairs ---

#[test]
fn all_1q_cliffords_are_unitary() {
    let identity = UnitaryMatrix::identity(2);
    for &cliff in Clifford::all_1q() {
        let mat = cliff.to_matrix();
        let product = mat.adjoint() * &mat;
        let diff = (product - identity.clone()).norm();
        assert!(
            diff < 1e-10,
            "Clifford::{cliff:?} is not unitary, diff = {diff}"
        );
    }
}

#[test]
fn all_2q_cliffords_are_unitary() {
    let identity = UnitaryMatrix::identity(4);
    for &cliff in Clifford::all_2q() {
        let mat = cliff.to_matrix();
        let product = mat.adjoint() * &mat;
        let diff = (product - identity.clone()).norm();
        assert!(
            diff < 1e-10,
            "Clifford::{cliff:?} is not unitary, diff = {diff}"
        );
    }
}

#[test]
fn clifford_1q_dagger_pairs_via_matrix() {
    let pairs = [
        (Clifford::SX, Clifford::SXdg),
        (Clifford::SY, Clifford::SYdg),
        (Clifford::SZ, Clifford::SZdg),
        (Clifford::F, Clifford::Fdg),
        (Clifford::F2, Clifford::F2dg),
        (Clifford::F3, Clifford::F3dg),
        (Clifford::F4, Clifford::F4dg),
    ];
    let identity = UnitaryMatrix::identity(2);
    for (gate, dagger) in pairs {
        let product = gate.to_matrix() * dagger.to_matrix();
        assert!(
            product.equiv_up_to_phase(&identity),
            "{gate:?} * {dagger:?} should be identity up to phase"
        );
    }
}

#[test]
fn clifford_2q_dagger_pairs_via_matrix() {
    let pairs = [
        (Clifford::SXX, Clifford::SXXdg),
        (Clifford::SYY, Clifford::SYYdg),
        (Clifford::SZZ, Clifford::SZZdg),
        (Clifford::ISWAP, Clifford::ISWAPdg),
        (Clifford::G, Clifford::Gdg),
    ];
    let identity = UnitaryMatrix::identity(4);
    for (gate, dagger) in pairs {
        let product = gate.to_matrix() * dagger.to_matrix();
        assert!(
            product.equiv_up_to_phase(&identity),
            "{gate:?} * {dagger:?} should be identity up to phase"
        );
    }
}

#[test]
fn self_adjoint_1q_cliffords_are_involutions_via_matrix() {
    let identity = UnitaryMatrix::identity(2);
    // All 1q self-adjoint Cliffords: Paulis + Hadamard variants
    let involutions = [
        Clifford::I,
        Clifford::X,
        Clifford::Y,
        Clifford::Z,
        Clifford::H,
        Clifford::H2,
        Clifford::H3,
        Clifford::H4,
        Clifford::H5,
        Clifford::H6,
    ];
    for cliff in involutions {
        let mat = cliff.to_matrix();
        let product = &mat * &mat;
        assert!(
            product.equiv_up_to_phase(&identity),
            "{cliff:?}^2 should be identity up to phase"
        );
    }
}

#[test]
fn self_adjoint_2q_cliffords_are_involutions_via_matrix() {
    let identity = UnitaryMatrix::identity(4);
    let involutions = [Clifford::CX, Clifford::CY, Clifford::CZ, Clifford::SWAP];
    for cliff in involutions {
        let mat = cliff.to_matrix();
        let product = &mat * &mat;
        assert!(
            product.equiv_up_to_phase(&identity),
            "{cliff:?}^2 should be identity up to phase"
        );
    }
}

#[test]
fn ccx_is_involution_via_matrix() {
    use pecos_core::gate_type::GateType;
    let mat = Unitary::Named(GateType::CCX).to_matrix();
    let identity = UnitaryMatrix::identity(8);
    let product = &mat * &mat;
    assert!(
        product.equiv_up_to_phase(&identity),
        "CCX^2 should be identity up to phase"
    );
}

// --- Rotation edge cases: zero angle and full turn ---

#[test]
fn rotation_zero_angle_is_identity() {
    let identity = UnitaryMatrix::identity(2);
    for rot in [RotationType::RX, RotationType::RY, RotationType::RZ] {
        let mat = Unitary::Rotation {
            rotation_type: rot,
            angle: Angle64::ZERO,
        }
        .to_matrix();
        let diff = (&mat - &identity).norm();
        assert!(diff < 1e-10, "{rot:?}(0) should be identity, diff = {diff}");
    }
}

#[test]
fn rotation_full_turn_is_identity() {
    let identity = UnitaryMatrix::identity(2);
    for rot in [RotationType::RX, RotationType::RY, RotationType::RZ] {
        let mat = Unitary::Rotation {
            rotation_type: rot,
            angle: Angle64::FULL_TURN,
        }
        .to_matrix();
        let diff = (&mat - &identity).norm();
        assert!(
            diff < 1e-10,
            "{rot:?}(2pi) should be identity, diff = {diff}"
        );
    }
}

#[test]
fn rotation_2q_zero_angle_is_identity() {
    let identity = UnitaryMatrix::identity(4);
    for rot in [RotationType::RXX, RotationType::RYY, RotationType::RZZ] {
        let u = Unitary::Rotation {
            rotation_type: rot,
            angle: Angle64::ZERO,
        };
        let mat = u.on_qubits(0, 1).to_matrix();
        let diff = (&mat - &identity).norm();
        assert!(diff < 1e-10, "{rot:?}(0) should be identity, diff = {diff}");
    }
}

// --- Rotation half-turn equals Pauli gates ---

#[test]
fn rotation_half_turn_equals_pauli_up_to_phase() {
    let rx_pi = Unitary::Rotation {
        rotation_type: RotationType::RX,
        angle: Angle64::HALF_TURN,
    }
    .to_matrix();
    let x = Pauli::X.to_matrix();
    assert!(
        rx_pi.equiv_up_to_phase(&x),
        "RX(pi) should equal X up to phase"
    );

    let ry_pi = Unitary::Rotation {
        rotation_type: RotationType::RY,
        angle: Angle64::HALF_TURN,
    }
    .to_matrix();
    let y = Pauli::Y.to_matrix();
    assert!(
        ry_pi.equiv_up_to_phase(&y),
        "RY(pi) should equal Y up to phase"
    );

    let rz_pi = Unitary::Rotation {
        rotation_type: RotationType::RZ,
        angle: Angle64::HALF_TURN,
    }
    .to_matrix();
    let z = Pauli::Z.to_matrix();
    assert!(
        rz_pi.equiv_up_to_phase(&z),
        "RZ(pi) should equal Z up to phase"
    );
}

// --- Algebraic property: dg anti-homomorphism (A*B).dg() == B.dg()*A.dg() ---

#[test]
fn dg_anti_homomorphism_via_matrix() {
    use pecos_core::unitary_rep;

    let a = unitary_rep::H(0);
    let b = unitary_rep::SZ(0);

    // (A * B).dg()
    let ab_dg = (a.clone() * b.clone()).dg().to_matrix();
    // B.dg() * A.dg()
    let bdg_adg = (b.dg() * a.dg()).to_matrix();

    assert!(
        ab_dg.equiv_up_to_phase(&bdg_adg),
        "(H*SZ).dg() should equal SZ.dg()*H.dg()"
    );
}

#[test]
fn dg_anti_homomorphism_2q_via_matrix() {
    use pecos_core::unitary_rep;

    let a = unitary_rep::CX(0, 1);
    let b = unitary_rep::H(0);

    let ab_dg = (a.clone() * b.clone()).dg().to_matrix();
    let bdg_adg = (b.dg() * a.dg()).to_matrix();

    assert!(
        ab_dg.equiv_up_to_phase(&bdg_adg),
        "(CX*H).dg() should equal H.dg()*CX.dg()"
    );
}

// --- Algebraic property: rotation angle additivity RZ(a)*RZ(b) == RZ(a+b) ---

#[test]
fn rotation_angle_additivity() {
    use pecos_core::unitary_rep::RZ;

    let a = Angle64::QUARTER_TURN;
    let b = Angle64::QUARTER_TURN;

    // RZ(a) * RZ(b) should equal RZ(a+b) = RZ(pi)
    let composed = (RZ(a, 0) * RZ(b, 0)).to_matrix();
    let direct = RZ(a + b, 0).to_matrix();

    assert!(
        composed.equiv_up_to_phase(&direct),
        "RZ(pi/2)*RZ(pi/2) should equal RZ(pi) up to phase"
    );
}

#[test]
fn rotation_angle_additivity_various() {
    use pecos_core::unitary_rep::{RX, RY};

    let q = Angle64::QUARTER_TURN;
    let h = Angle64::HALF_TURN;
    let tq = Angle64::THREE_QUARTERS_TURN;

    // RX(pi/2) * RX(pi) = RX(3pi/2)
    let composed = (RX(q, 0) * RX(h, 0)).to_matrix();
    let direct = RX(q + h, 0).to_matrix();
    assert!(
        composed.equiv_up_to_phase(&direct),
        "RX(pi/2)*RX(pi) should equal RX(3pi/2)"
    );

    // RY(pi/2) * RY(3pi/2) = RY(2pi) = I
    let composed = (RY(q, 0) * RY(tq, 0)).to_matrix();
    let identity = UnitaryMatrix::identity(2);
    assert!(
        composed.equiv_up_to_phase(&identity),
        "RY(pi/2)*RY(3pi/2) should equal identity"
    );
}

// --- Algebraic property: RX(-theta) == RX(theta).adjoint() ---

#[test]
fn rotation_adjoint_negates_angle_via_matrix() {
    use pecos_core::unitary_rep::{RX, RY, RZ};

    let angle = Angle64::QUARTER_TURN;
    let neg_angle = Angle64::THREE_QUARTERS_TURN;

    // R(-theta) = R(theta)^dagger up to global phase
    // (THREE_QUARTERS_TURN and -QUARTER_TURN differ by 2pi but the half-angle
    //  causes an overall -1 factor, so we check equiv_up_to_phase)
    for (make_rot, name) in [
        (RX as fn(Angle64, usize) -> pecos_core::UnitaryRep, "RX"),
        (RY as fn(Angle64, usize) -> pecos_core::UnitaryRep, "RY"),
        (RZ as fn(Angle64, usize) -> pecos_core::UnitaryRep, "RZ"),
    ] {
        let mat = make_rot(angle, 0).to_matrix();
        let mat_neg = make_rot(neg_angle, 0).to_matrix();
        let mat_adj = mat.adjoint();

        assert!(
            mat_neg.equiv_up_to_phase(&mat_adj),
            "{name}(-theta) should equal {name}(theta).adjoint() up to phase"
        );
    }
}

// --- Consistency: rotation_to_gate_type agrees with gate_to_matrix ---

#[test]
fn rotation_matches_named_gate_when_gate_type_exists() {
    use pecos_core::gate_type::GateType;
    use pecos_core::unitary_rep::rotation_to_gate_type;

    let cases: &[(RotationType, Angle64, GateType)] = &[
        (RotationType::RX, Angle64::QUARTER_TURN, GateType::SX),
        (
            RotationType::RX,
            Angle64::THREE_QUARTERS_TURN,
            GateType::SXdg,
        ),
        (RotationType::RX, Angle64::HALF_TURN, GateType::X),
        (RotationType::RY, Angle64::QUARTER_TURN, GateType::SY),
        (
            RotationType::RY,
            Angle64::THREE_QUARTERS_TURN,
            GateType::SYdg,
        ),
        (RotationType::RY, Angle64::HALF_TURN, GateType::Y),
        (RotationType::RZ, Angle64::QUARTER_TURN, GateType::SZ),
        (
            RotationType::RZ,
            Angle64::THREE_QUARTERS_TURN,
            GateType::SZdg,
        ),
        (RotationType::RZ, Angle64::HALF_TURN, GateType::Z),
    ];

    for &(rot, angle, expected_gt) in cases {
        // Verify rotation_to_gate_type returns the expected gate
        assert_eq!(
            rotation_to_gate_type(rot, angle),
            Some(expected_gt),
            "{rot:?}({angle:?}) should map to {expected_gt:?}"
        );

        // Verify the rotation matrix matches the named gate matrix
        let rot_mat = Unitary::Rotation {
            rotation_type: rot,
            angle,
        }
        .to_matrix();
        let named_mat = Unitary::Named(expected_gt).to_matrix();
        assert!(
            rot_mat.equiv_up_to_phase(&named_mat),
            "{rot:?}({angle:?}) matrix should match Named({expected_gt:?}) matrix"
        );
    }
}

#[test]
fn rotation_2q_matches_named_gate_when_gate_type_exists() {
    use pecos_core::gate_type::GateType;

    let cases: &[(RotationType, Angle64, GateType)] = &[
        (RotationType::RXX, Angle64::QUARTER_TURN, GateType::SXX),
        (
            RotationType::RXX,
            Angle64::THREE_QUARTERS_TURN,
            GateType::SXXdg,
        ),
        (RotationType::RYY, Angle64::QUARTER_TURN, GateType::SYY),
        (
            RotationType::RYY,
            Angle64::THREE_QUARTERS_TURN,
            GateType::SYYdg,
        ),
        (RotationType::RZZ, Angle64::QUARTER_TURN, GateType::SZZ),
        (
            RotationType::RZZ,
            Angle64::THREE_QUARTERS_TURN,
            GateType::SZZdg,
        ),
    ];

    for &(rot, angle, expected_gt) in cases {
        let rot_mat = Unitary::Rotation {
            rotation_type: rot,
            angle,
        }
        .on_qubits(0, 1)
        .to_matrix();
        let named_mat = Unitary::Named(expected_gt).on_qubits(0, 1).to_matrix();
        assert!(
            rot_mat.equiv_up_to_phase(&named_mat),
            "{rot:?}({angle:?}) matrix should match Named({expected_gt:?}) matrix"
        );
    }
}

// --- Qubit embedding: gates act on correct qubits ---

#[test]
fn qubit_embedding_non_default_indices() {
    // X on qubit 2 in a 3-qubit system should differ from X on qubit 0
    let x_q0 = pecos_core::unitary_rep::X(0);
    let x_q2 = pecos_core::unitary_rep::X(2);

    let mat_q0 = pecos_quantum::unitary_matrix::to_matrix_with_size(&x_q0, 3);
    let mat_q2 = pecos_quantum::unitary_matrix::to_matrix_with_size(&x_q2, 3);

    assert!(!mat_q0.equiv_up_to_phase(&mat_q2));
}

#[test]
fn cx_qubit_order_matters() {
    // CX(0,1) != CX(1,0)
    let cx_01 = pecos_core::unitary_rep::CX(0, 1).to_matrix();
    let cx_10 = pecos_core::unitary_rep::CX(1, 0).to_matrix();
    assert!(
        !cx_01.equiv_up_to_phase(&cx_10),
        "CX(0,1) should differ from CX(1,0)"
    );
}

// --- Face gate cycle: F^3 = I (up to phase) ---

#[test]
fn face_gates_have_order_3() {
    let identity = UnitaryMatrix::identity(2);
    for cliff in [
        Clifford::F,
        Clifford::Fdg,
        Clifford::F2,
        Clifford::F2dg,
        Clifford::F3,
        Clifford::F3dg,
        Clifford::F4,
        Clifford::F4dg,
    ] {
        let mat = cliff.to_matrix();
        let cubed = &(&mat * &mat) * &mat;
        assert!(
            cubed.equiv_up_to_phase(&identity),
            "{cliff:?}^3 should be identity up to phase"
        );
    }
}

// --- Exact matrix values for standard gates ---

#[test]
fn exact_matrix_values_pauli_y() {
    let mat = Pauli::Y.to_matrix();
    let zero = Complex64::new(0.0, 0.0);
    let i = Complex64::new(0.0, 1.0);
    let neg_i = Complex64::new(0.0, -1.0);
    assert!((mat[(0, 0)] - zero).norm() < 1e-10);
    assert!((mat[(0, 1)] - neg_i).norm() < 1e-10);
    assert!((mat[(1, 0)] - i).norm() < 1e-10);
    assert!((mat[(1, 1)] - zero).norm() < 1e-10);
}

#[test]
fn exact_matrix_values_pauli_z() {
    let mat = Pauli::Z.to_matrix();
    let one = Complex64::new(1.0, 0.0);
    let neg_one = Complex64::new(-1.0, 0.0);
    assert!((mat[(0, 0)] - one).norm() < 1e-10);
    assert!((mat[(0, 1)]).norm() < 1e-10);
    assert!((mat[(1, 0)]).norm() < 1e-10);
    assert!((mat[(1, 1)] - neg_one).norm() < 1e-10);
}

#[test]
fn exact_matrix_values_hadamard() {
    let mat = Clifford::H.to_matrix();
    let s = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
    assert!((mat[(0, 0)] - s).norm() < 1e-10);
    assert!((mat[(0, 1)] - s).norm() < 1e-10);
    assert!((mat[(1, 0)] - s).norm() < 1e-10);
    assert!((mat[(1, 1)] + s).norm() < 1e-10); // -1/sqrt(2)
}

#[test]
fn hadamard_squared_is_identity_exact() {
    let h = Clifford::H.to_matrix();
    let h2 = &h * &h;
    let identity = UnitaryMatrix::identity(2);
    let diff = (&h2 - &identity).norm();
    assert!(
        diff < 1e-10,
        "H^2 should be exactly identity, diff = {diff}"
    );
}

#[test]
fn s_gates_have_order_4() {
    let identity = UnitaryMatrix::identity(2);
    for cliff in [
        Clifford::SX,
        Clifford::SXdg,
        Clifford::SY,
        Clifford::SYdg,
        Clifford::SZ,
        Clifford::SZdg,
    ] {
        let mat = cliff.to_matrix();
        let sq = &mat * &mat;
        let fourth = &sq * &sq;
        assert!(
            fourth.equiv_up_to_phase(&identity),
            "{cliff:?}^4 should be identity up to phase"
        );
    }
}

#[test]
fn t_gate_has_order_8() {
    use pecos_core::gate_type::GateType;
    let identity = UnitaryMatrix::identity(2);
    for gt in [GateType::T, GateType::Tdg] {
        let mat = Unitary::Named(gt).to_matrix();
        let sq = &mat * &mat;
        let fourth = &sq * &sq;
        let eighth = &fourth * &fourth;
        assert!(
            eighth.equiv_up_to_phase(&identity),
            "{gt:?}^8 should be identity up to phase"
        );
    }
}

#[test]
fn s_gate_squared_is_pauli() {
    // SX^2 = X, SY^2 = Y, SZ^2 = Z (up to phase)
    let pairs = [
        (Clifford::SX, Pauli::X),
        (Clifford::SY, Pauli::Y),
        (Clifford::SZ, Pauli::Z),
    ];
    for (s_gate, pauli) in pairs {
        let mat = s_gate.to_matrix();
        let sq = &mat * &mat;
        let pauli_mat = pauli.to_matrix();
        assert!(
            sq.equiv_up_to_phase(&pauli_mat),
            "{s_gate:?}^2 should equal {pauli:?} up to phase"
        );
    }
}

// --- Every Clifford: matrix adjoint matches dg() variant's matrix ---

#[test]
fn all_1q_clifford_adjoint_matches_inverse_matrix() {
    for &cliff in Clifford::all_1q() {
        let mat = cliff.to_matrix();
        let mat_adj = mat.adjoint();
        let inv_mat = cliff.inverse().to_matrix();
        assert!(
            mat_adj.equiv_up_to_phase(&inv_mat),
            "Clifford::{cliff:?}.to_matrix().adjoint() should match {cliff:?}.inverse().to_matrix()"
        );
    }
}

#[test]
fn all_2q_clifford_adjoint_matches_inverse_matrix() {
    for &cliff in Clifford::all_2q() {
        let mat = cliff.to_matrix();
        let mat_adj = mat.adjoint();
        let inv_mat = cliff.inverse().to_matrix();
        assert!(
            mat_adj.equiv_up_to_phase(&inv_mat),
            "Clifford::{cliff:?}.to_matrix().adjoint() should match {cliff:?}.inverse().to_matrix()"
        );
    }
}

// --- Exact 2-qubit matrix values ---

#[test]
fn exact_matrix_values_cx() {
    let mat = Clifford::CX.to_matrix();
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    // CX(control=q0, target=q1) in little-endian basis (q0=LSB):
    // |00>->|00>, |01>->|11>, |10>->|10>, |11>->|01>
    // where index = q0 + 2*q1
    let expected = UnitaryMatrix::from(nalgebra::DMatrix::from_row_slice(
        4,
        4,
        &[
            one, zero, zero, zero, zero, zero, zero, one, zero, zero, one, zero, zero, one, zero,
            zero,
        ],
    ));
    assert!(
        mat.equiv_up_to_phase(&expected),
        "CX matrix does not match expected"
    );
}

#[test]
fn exact_matrix_values_cz() {
    let mat = Clifford::CZ.to_matrix();
    let one = Complex64::new(1.0, 0.0);
    let neg = Complex64::new(-1.0, 0.0);
    // CZ = diag(1, 1, 1, -1) (same in both big-endian and little-endian)
    let expected = UnitaryMatrix::diag(&[one, one, one, neg]);
    assert!(
        mat.equiv_up_to_phase(&expected),
        "CZ matrix does not match expected"
    );
}

#[test]
fn exact_matrix_values_swap() {
    let mat = Clifford::SWAP.to_matrix();
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    // SWAP in little-endian: swaps indices 1 and 2
    // |00>->|00>, |01>->|10>, |10>->|01>, |11>->|11>
    let expected = UnitaryMatrix::from(nalgebra::DMatrix::from_row_slice(
        4,
        4,
        &[
            one, zero, zero, zero, zero, zero, one, zero, zero, one, zero, zero, zero, zero, zero,
            one,
        ],
    ));
    assert!(
        mat.equiv_up_to_phase(&expected),
        "SWAP matrix does not match expected"
    );
}

/// Verify that the iSWAP matrix correctly implements the standard definition:
/// iSWAP = exp(i*pi/4*(XX+YY)), which maps |01> -> i|10>, |10> -> i|01>.
#[test]
fn exact_matrix_values_iswap() {
    let mat = op::ISWAP(0, 1).to_matrix();
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    let i = Complex64::new(0.0, 1.0);
    // Standard iSWAP in little-endian basis (same as big-endian since SWAP is symmetric):
    // [[1,0,0,0],[0,0,i,0],[0,i,0,0],[0,0,0,1]]
    let expected = UnitaryMatrix::from(nalgebra::DMatrix::from_row_slice(
        4,
        4,
        &[
            one, zero, zero, zero, zero, zero, i, zero, zero, i, zero, zero, zero, zero, zero, one,
        ],
    ));
    assert!(
        mat.equiv_up_to_phase(&expected),
        "iSWAP matrix does not match standard definition\nActual:\n{mat}"
    );
}

/// Verify that iSWAPdg matches the standard iSWAP† (via Op path).
#[test]
fn exact_matrix_values_iswapdg() {
    let mat = op::ISWAPdg(0, 1).to_matrix();
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    let neg_i = Complex64::new(0.0, -1.0);
    // iSWAP† = [[1,0,0,0],[0,0,-i,0],[0,-i,0,0],[0,0,0,1]]
    let expected = UnitaryMatrix::from(nalgebra::DMatrix::from_row_slice(
        4,
        4,
        &[
            one, zero, zero, zero, zero, zero, neg_i, zero, zero, neg_i, zero, zero, zero, zero,
            zero, one,
        ],
    ));
    assert!(
        mat.equiv_up_to_phase(&expected),
        "iSWAPdg matrix does not match standard definition\nActual:\n{mat}"
    );
}

/// Verify that `ISWAPdg` via the Clifford path also matches the standard iSWAP†.
#[test]
fn exact_matrix_values_iswapdg_clifford_path() {
    let mat = Clifford::ISWAPdg.to_matrix();
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    let neg_i = Complex64::new(0.0, -1.0);
    let expected = UnitaryMatrix::from(nalgebra::DMatrix::from_row_slice(
        4,
        4,
        &[
            one, zero, zero, zero, zero, zero, neg_i, zero, zero, neg_i, zero, zero, zero, zero,
            zero, one,
        ],
    ));
    assert!(
        mat.equiv_up_to_phase(&expected),
        "ISWAPdg (Clifford path) does not match standard definition\nActual:\n{mat}"
    );
}

/// Verify that Gdg via the Clifford path matches G (since G is self-inverse).
#[test]
fn exact_matrix_values_gdg_clifford_path() {
    let g_mat = Clifford::G.to_matrix();
    let gdg_mat = Clifford::Gdg.to_matrix();
    assert!(
        g_mat.equiv_up_to_phase(&gdg_mat),
        "G and Gdg should be equivalent (G is self-inverse)\nG:\n{g_mat}\nGdg:\n{gdg_mat}"
    );
}

// --- Pauli transformation verification for 1-qubit Hadamard and Face gates ---

/// Verify that U maps Pauli `input` to `sign * output` via conjugation: U P U† = s * Q.
fn assert_pauli_transform(
    mat: &UnitaryMatrix,
    input: &UnitaryMatrix,
    output: &UnitaryMatrix,
    sign: f64,
    name: &str,
    pauli_name: &str,
) {
    let mat_adj = mat.adjoint();
    let result = &(mat * &(input * &mat_adj)) * Complex64::new(sign, 0.0);
    let diff = (&result - output).norm();
    assert!(
        diff < 1e-10,
        "{name}: {pauli_name} transformation failed, diff = {diff}"
    );
}

fn pauli_x() -> UnitaryMatrix {
    UnitaryMatrix::from(nalgebra::DMatrix::from_row_slice(
        2,
        2,
        &[
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    ))
}

fn pauli_y() -> UnitaryMatrix {
    UnitaryMatrix::from(nalgebra::DMatrix::from_row_slice(
        2,
        2,
        &[
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 0.0),
        ],
    ))
}

fn pauli_z() -> UnitaryMatrix {
    UnitaryMatrix::from(nalgebra::DMatrix::from_row_slice(
        2,
        2,
        &[
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ],
    ))
}

#[test]
fn hadamard_variants_pauli_transformations() {
    let x = pauli_x();
    let y = pauli_y();
    let z = pauli_z();

    // H2: X -> -Z, Y -> -Y, Z -> -X
    let h2 = Clifford::H2.to_matrix();
    assert_pauli_transform(&h2, &x, &z, -1.0, "H2", "X->-Z");
    assert_pauli_transform(&h2, &y, &y, -1.0, "H2", "Y->-Y");
    assert_pauli_transform(&h2, &z, &x, -1.0, "H2", "Z->-X");

    // H3: X -> Y, Y -> X, Z -> -Z
    let h3 = Clifford::H3.to_matrix();
    assert_pauli_transform(&h3, &x, &y, 1.0, "H3", "X->Y");
    assert_pauli_transform(&h3, &y, &x, 1.0, "H3", "Y->X");
    assert_pauli_transform(&h3, &z, &z, -1.0, "H3", "Z->-Z");

    // H4: X -> -Y, Y -> -X, Z -> -Z
    let h4 = Clifford::H4.to_matrix();
    assert_pauli_transform(&h4, &x, &y, -1.0, "H4", "X->-Y");
    assert_pauli_transform(&h4, &y, &x, -1.0, "H4", "Y->-X");
    assert_pauli_transform(&h4, &z, &z, -1.0, "H4", "Z->-Z");

    // H5: X -> -X, Y -> Z, Z -> Y
    let h5 = Clifford::H5.to_matrix();
    assert_pauli_transform(&h5, &x, &x, -1.0, "H5", "X->-X");
    assert_pauli_transform(&h5, &y, &z, 1.0, "H5", "Y->Z");
    assert_pauli_transform(&h5, &z, &y, 1.0, "H5", "Z->Y");

    // H6: X -> -X, Y -> -Z, Z -> -Y
    let h6 = Clifford::H6.to_matrix();
    assert_pauli_transform(&h6, &x, &x, -1.0, "H6", "X->-X");
    assert_pauli_transform(&h6, &y, &z, -1.0, "H6", "Y->-Z");
    assert_pauli_transform(&h6, &z, &y, -1.0, "H6", "Z->-Y");
}

#[test]
fn face_gates_pauli_transformations() {
    let x = pauli_x();
    let y = pauli_y();
    let z = pauli_z();

    // Check F and Fdg are mutual inverses via Pauli transforms
    // F: known to have order 3 (tested elsewhere), check X/Y/Z images
    let f = Clifford::F.to_matrix();
    // Verify F*X*F† etc. against CliffordRep
    let f_rep = CliffordRep::f(0);
    verify_1q_pauli_transforms(&f, &f_rep, "F", &x, &y, &z);

    let fdg = Clifford::Fdg.to_matrix();
    let fdg_rep = CliffordRep::fdg(0);
    verify_1q_pauli_transforms(&fdg, &fdg_rep, "Fdg", &x, &y, &z);

    for cliff in [
        Clifford::F2,
        Clifford::F2dg,
        Clifford::F3,
        Clifford::F3dg,
        Clifford::F4,
        Clifford::F4dg,
    ] {
        let mat = cliff.to_matrix();
        let rep = cliff.on_qubit(0);
        verify_1q_pauli_transforms(&mat, &rep, &format!("{cliff}"), &x, &y, &z);
    }
}

/// Verify that a matrix correctly implements the Pauli transformations
/// defined by a `CliffordRep`, for a single-qubit gate.
fn verify_1q_pauli_transforms(
    mat: &UnitaryMatrix,
    rep: &CliffordRep,
    name: &str,
    x: &UnitaryMatrix,
    _y: &UnitaryMatrix,
    z: &UnitaryMatrix,
) {
    let mat_adj = mat.adjoint();
    let x_image = rep.x_image(0);
    let z_image = rep.z_image(0);
    let inputs: [(&str, &UnitaryMatrix, &PauliString); 2] = [("X", x, x_image), ("Z", z, z_image)];

    for (in_name, in_mat, image) in &inputs {
        // Compute U * P * U†
        let result = mat * &(*in_mat * &mat_adj.clone());
        // Build expected Pauli matrix from the CliffordRep image
        let expected = pauli_string_to_1q_matrix(image);
        let diff = (&result - &expected).norm();
        assert!(
            diff < 1e-10,
            "{name}: {in_name} transformation failed, diff = {diff}"
        );
    }
}

/// Convert a single-qubit `PauliString` to its 2x2 matrix representation.
fn pauli_string_to_1q_matrix(ps: &PauliString) -> UnitaryMatrix {
    let x = pauli_x();
    let y = pauli_y();
    let z = pauli_z();
    let i = UnitaryMatrix::identity(2);

    let pauli = ps.get(0);
    let base = match pauli {
        Pauli::I => i,
        Pauli::X => x,
        Pauli::Y => y,
        Pauli::Z => z,
    };
    let phase: Complex64 = ps.phase().to_complex();
    &base * phase
}

// --- Exact 2-qubit matrix values for G gate ---

#[test]
fn exact_matrix_values_g() {
    // G = 1/2 [[1, 1, 1, -1], [1, -1, 1, 1], [1, 1, -1, 1], [-1, 1, 1, 1]]
    let mat = Clifford::G.to_matrix();
    let h = Complex64::new(0.5, 0.0);
    let nh = Complex64::new(-0.5, 0.0);
    let expected = UnitaryMatrix::from(nalgebra::DMatrix::from_row_slice(
        4,
        4,
        &[h, h, h, nh, h, nh, h, h, h, h, nh, h, nh, h, h, h],
    ));
    assert!(
        mat.equiv_up_to_phase(&expected),
        "G matrix does not match expected\nActual:\n{mat}"
    );
}

#[test]
fn g_squared_is_identity() {
    // G is self-inverse: G^2 = I
    let g = Clifford::G.to_matrix();
    let g2 = &g * &g;
    let identity = UnitaryMatrix::identity(4);
    let diff = (&g2 - &identity).norm();
    assert!(diff < 1e-10, "G^2 should be identity, diff = {diff}");
}

// --- 2-qubit Pauli conjugation: U * P * U† matches CliffordRep images ---

/// Build a 2-qubit Pauli matrix from a `PauliString` (phase * P0 tensor P1).
fn pauli_string_to_2q_matrix(ps: &PauliString) -> UnitaryMatrix {
    let i2 = UnitaryMatrix::identity(2);
    let px = pauli_x();
    let py = pauli_y();
    let pz = pauli_z();

    let mat = |p: Pauli| -> &UnitaryMatrix {
        match p {
            Pauli::I => &i2,
            Pauli::X => &px,
            Pauli::Y => &py,
            Pauli::Z => &pz,
        }
    };

    // Little-endian: index = q0 + 2*q1, so tensor order is q1 tensor q0
    let base = mat(ps.get(1)) & mat(ps.get(0));
    let phase: Complex64 = ps.phase().to_complex();
    &base * phase
}

#[test]
fn two_qubit_clifford_pauli_conjugation() {
    // For each 2q Clifford gate, verify U * P * U† matches the CliffordRep
    // Pauli images for all 4 generators: X0, X1, Z0, Z1.
    for &cliff in Clifford::all_2q() {
        let mat = cliff.to_matrix();
        let mat_adj = mat.adjoint();
        let rep = cliff.on_qubits(0, 1);

        let i2 = UnitaryMatrix::identity(2);
        // Little-endian: index = q0 + 2*q1, so matrix is q1_pauli tensor q0_pauli
        let generators: [(usize, &str, UnitaryMatrix); 4] = [
            (0, "X0", &i2 & pauli_x()), // I tensor X = X on q0 (LSB)
            (1, "X1", pauli_x() & &i2), // X tensor I = X on q1 (MSB)
            (0, "Z0", &i2 & pauli_z()), // I tensor Z = Z on q0
            (1, "Z1", pauli_z() & &i2), // Z tensor I = Z on q1
        ];

        for (qubit, gen_name, gen_mat) in &generators {
            // Compute U * P * U†
            let result = &mat * &(gen_mat * &mat_adj.clone());

            // Get expected image from CliffordRep
            let image = if gen_name.starts_with('X') {
                rep.x_image(*qubit)
            } else {
                rep.z_image(*qubit)
            };
            let expected = pauli_string_to_2q_matrix(image);

            let diff = (&result - &expected).norm();
            assert!(
                diff < 1e-10,
                "{cliff}: {gen_name} conjugation failed, diff = {diff}"
            );
        }
    }
}

// --- Op-level CliffordRep/UnitaryRep dual consistency ---

#[test]
fn op_clifford_rep_matches_unitary_rep_2q() {
    // For each 2q gate constructor, verify the CliffordRep and UnitaryRep
    // halves stored inside Op::Clifford produce the same matrix.
    let ops: Vec<(&str, Op)> = vec![
        ("CX", op::CX(0, 1)),
        ("CY", op::CY(0, 1)),
        ("CZ", op::CZ(0, 1)),
        ("SWAP", op::SWAP(0, 1)),
        ("SXX", op::SXX(0, 1)),
        ("SXXdg", op::SXXdg(0, 1)),
        ("SYY", op::SYY(0, 1)),
        ("SYYdg", op::SYYdg(0, 1)),
        ("SZZ", op::SZZ(0, 1)),
        ("SZZdg", op::SZZdg(0, 1)),
        ("ISWAP", op::ISWAP(0, 1)),
        ("ISWAPdg", op::ISWAPdg(0, 1)),
        ("G", op::G(0, 1)),
        ("Gdg", op::Gdg(0, 1)),
    ];

    for (name, o) in ops {
        let cr_mat = o.as_clifford().unwrap().to_matrix();
        let ur_mat = o.into_unitary().unwrap().to_matrix();
        assert!(
            cr_mat.equiv_up_to_phase(&ur_mat),
            "{name}: CliffordRep and UnitaryRep matrices disagree\nCR:\n{cr_mat}\nUR:\n{ur_mat}"
        );
    }
}

#[test]
fn op_clifford_rep_matches_unitary_rep_1q() {
    // Note: I, X, Y, Z are Pauli-level ops (Op::Pauli), not Clifford, so they
    // don't have a CliffordRep half. Only test Clifford-level ops here.
    let ops: Vec<(&str, Op)> = vec![
        ("H", op::H(0)),
        ("SX", op::SX(0)),
        ("SXdg", op::SXdg(0)),
        ("SY", op::SY(0)),
        ("SYdg", op::SYdg(0)),
        ("SZ", op::SZ(0)),
        ("SZdg", op::SZdg(0)),
        ("H2", op::H2(0)),
        ("H3", op::H3(0)),
        ("H4", op::H4(0)),
        ("H5", op::H5(0)),
        ("H6", op::H6(0)),
        ("F", op::F(0)),
        ("Fdg", op::Fdg(0)),
        ("F2", op::F2(0)),
        ("F2dg", op::F2dg(0)),
        ("F3", op::F3(0)),
        ("F3dg", op::F3dg(0)),
        ("F4", op::F4(0)),
        ("F4dg", op::F4dg(0)),
    ];

    for (name, o) in ops {
        let cr_mat = o.as_clifford().unwrap().to_matrix();
        let ur_mat = o.into_unitary().unwrap().to_matrix();
        assert!(
            cr_mat.equiv_up_to_phase(&ur_mat),
            "{name}: CliffordRep and UnitaryRep matrices disagree\nCR:\n{cr_mat}\nUR:\n{ur_mat}"
        );
    }
}

// --- Op composition consistency: CliffordRep and UnitaryRep halves agree after * ---

use pecos_core::op::Op;

/// Verify that a `CliffordRep` and a `UnitaryMatrix` agree on their Pauli images.
/// This avoids calling `CliffordRep::to_matrix()` which can't handle arbitrary compositions.
fn assert_clifford_rep_matches_matrix_1q(cr: &CliffordRep, mat: &UnitaryMatrix, label: &str) {
    let mat_adj = mat.adjoint();
    let x = pauli_x();
    let z = pauli_z();

    for (gen_name, gen_mat, image) in [("X", &x, cr.x_image(0)), ("Z", &z, cr.z_image(0))] {
        let result = mat * &(gen_mat * &mat_adj.clone());
        let expected = pauli_string_to_1q_matrix(image);
        let diff = (&result - &expected).norm();
        assert!(
            diff < 1e-10,
            "{label}: {gen_name} image mismatch, diff = {diff}"
        );
    }
}

fn assert_clifford_rep_matches_matrix_2q(cr: &CliffordRep, mat: &UnitaryMatrix, label: &str) {
    let mat_adj = mat.adjoint();
    let i2 = UnitaryMatrix::identity(2);
    let generators: [(usize, &str, UnitaryMatrix); 4] = [
        (0, "X0", &i2 & pauli_x()),
        (1, "X1", pauli_x() & &i2),
        (0, "Z0", &i2 & pauli_z()),
        (1, "Z1", pauli_z() & &i2),
    ];

    for (qubit, gen_name, gen_mat) in &generators {
        let result = mat * &(gen_mat * &mat_adj.clone());
        let image = if gen_name.starts_with('X') {
            cr.x_image(*qubit)
        } else {
            cr.z_image(*qubit)
        };
        let expected = pauli_string_to_2q_matrix(image);
        let diff = (&result - &expected).norm();
        assert!(
            diff < 1e-10,
            "{label}: {gen_name} image mismatch, diff = {diff}"
        );
    }
}

#[test]
fn op_composition_preserves_dual_consistency_1q() {
    // Compose pairs of 1q Clifford Ops and verify CliffordRep and UnitaryRep agree.
    // Only use Clifford-level ops (not Pauli-level X/Y/Z).
    let gates: Vec<Op> = vec![
        op::H(0),
        op::SZ(0),
        op::SX(0),
        op::F(0),
        op::H2(0),
        op::SY(0),
    ];

    for a in &gates {
        for b in &gates {
            let composed = a.clone() * b.clone();
            let cr = composed.as_clifford().unwrap();
            let ur_mat = composed.clone().into_unitary().unwrap().to_matrix();
            assert_clifford_rep_matches_matrix_1q(cr, &ur_mat, &format!("{a} * {b}"));
        }
    }
}

#[test]
fn op_composition_preserves_dual_consistency_2q() {
    // Compose pairs of 2q Clifford Ops and verify CliffordRep and UnitaryRep agree.
    let gates: Vec<Op> = vec![
        op::CX(0, 1),
        op::CZ(0, 1),
        op::SWAP(0, 1),
        op::ISWAP(0, 1),
        op::G(0, 1),
    ];

    for a in &gates {
        for b in &gates {
            let composed = a.clone() * b.clone();
            let cr = composed.as_clifford().unwrap();
            let ur_mat = composed.clone().into_unitary().unwrap().to_matrix();
            assert_clifford_rep_matches_matrix_2q(cr, &ur_mat, &format!("{a} * {b}"));
        }
    }
}

#[test]
fn display_does_not_panic() {
    let mat = Pauli::X.to_matrix();
    let s = format!("{mat}");
    assert!(!s.is_empty());
}
