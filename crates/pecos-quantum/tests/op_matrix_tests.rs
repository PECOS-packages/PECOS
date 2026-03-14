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

//! Matrix-level verification that Op's dual representations are consistent.
//!
//! For each Clifford gate, the Op stores both a CliffordRep (tableau) and a
//! UnitaryRep (expression). This test verifies that the UnitaryRep expression
//! produces a matrix equivalent (up to global phase) to the expected gate.

use nalgebra::DMatrix;
use num_complex::Complex64;
use pecos_core::op;
use pecos_core::Angle64;
use pecos_quantum::unitary_matrix::{operators_equiv, ToMatrix};

/// Verify a 1-qubit Clifford Op's UnitaryRep matches a reference UnitaryRep.
fn check_1q_clifford(gate: pecos_core::Op, reference: pecos_core::UnitaryRep, name: &str) {
    let ur = gate.into_unitary().unwrap();
    assert!(
        operators_equiv(&ur, &reference),
        "{name}: UnitaryRep decomposition does not match reference"
    );
}

/// Verify a 2-qubit Clifford Op's UnitaryRep by checking unitarity and
/// that the matrix matches a reference.
fn check_2q_clifford(gate: pecos_core::Op, reference: pecos_core::UnitaryRep, name: &str) {
    let ur = gate.into_unitary().unwrap();
    assert!(
        operators_equiv(&ur, &reference),
        "{name}: UnitaryRep decomposition does not match reference"
    );
}

// ============================================================================
// 1-qubit Clifford gates with direct UnitaryRep references
// ============================================================================

#[test]
fn op_h_matches_unitary_rep() {
    check_1q_clifford(op::H(0), pecos_core::unitary_rep::H(0), "H");
}

#[test]
fn op_sx_matches_unitary_rep() {
    check_1q_clifford(op::SX(0), pecos_core::unitary_rep::SX(0), "SX");
}

#[test]
fn op_sxdg_matches_unitary_rep() {
    check_1q_clifford(op::SXdg(0), pecos_core::unitary_rep::SX(0).dg(), "SXdg");
}

#[test]
fn op_sy_matches_unitary_rep() {
    check_1q_clifford(op::SY(0), pecos_core::unitary_rep::SY(0), "SY");
}

#[test]
fn op_sydg_matches_unitary_rep() {
    check_1q_clifford(op::SYdg(0), pecos_core::unitary_rep::SY(0).dg(), "SYdg");
}

#[test]
fn op_sz_matches_unitary_rep() {
    check_1q_clifford(op::SZ(0), pecos_core::unitary_rep::SZ(0), "SZ");
}

#[test]
fn op_szdg_matches_unitary_rep() {
    check_1q_clifford(op::SZdg(0), pecos_core::unitary_rep::SZ(0).dg(), "SZdg");
}

// ============================================================================
// Hadamard variants — decomposition correctness
// ============================================================================

#[test]
fn op_h2_decomposition_correct() {
    // H2 = SY * Z
    let reference = pecos_core::unitary_rep::SY(0) * pecos_core::unitary_rep::Z(0);
    check_1q_clifford(op::H2(0), reference, "H2");
}

#[test]
fn op_h3_decomposition_correct() {
    // H3 = SZ * Y
    let reference = pecos_core::unitary_rep::SZ(0) * pecos_core::unitary_rep::Y(0);
    check_1q_clifford(op::H3(0), reference, "H3");
}

#[test]
fn op_h4_decomposition_correct() {
    // H4 = SZ * X
    let reference = pecos_core::unitary_rep::SZ(0) * pecos_core::unitary_rep::X(0);
    check_1q_clifford(op::H4(0), reference, "H4");
}

#[test]
fn op_h5_decomposition_correct() {
    // H5 = SX * Z
    let reference = pecos_core::unitary_rep::SX(0) * pecos_core::unitary_rep::Z(0);
    check_1q_clifford(op::H5(0), reference, "H5");
}

#[test]
fn op_h6_decomposition_correct() {
    // H6 = SX * Y
    let reference = pecos_core::unitary_rep::SX(0) * pecos_core::unitary_rep::Y(0);
    check_1q_clifford(op::H6(0), reference, "H6");
}

// ============================================================================
// Face gate variants — decomposition correctness
// ============================================================================

#[test]
fn op_f_decomposition_correct() {
    // F = SX * SZ
    let reference = pecos_core::unitary_rep::SX(0) * pecos_core::unitary_rep::SZ(0);
    check_1q_clifford(op::F(0), reference, "F");
}

#[test]
fn op_fdg_decomposition_correct() {
    let reference = (pecos_core::unitary_rep::SX(0) * pecos_core::unitary_rep::SZ(0)).dg();
    // F† should be equivalent to Fdg reference
    check_1q_clifford(op::Fdg(0), reference, "Fdg");
}

#[test]
fn op_f2_decomposition_correct() {
    let reference = pecos_core::unitary_rep::SX(0).dg() * pecos_core::unitary_rep::SY(0);
    check_1q_clifford(op::F2(0), reference, "F2");
}

#[test]
fn op_f2dg_decomposition_correct() {
    let reference = pecos_core::unitary_rep::SY(0).dg() * pecos_core::unitary_rep::SX(0);
    check_1q_clifford(op::F2dg(0), reference, "F2dg");
}

#[test]
fn op_f3_decomposition_correct() {
    let reference = pecos_core::unitary_rep::SX(0).dg() * pecos_core::unitary_rep::SZ(0);
    check_1q_clifford(op::F3(0), reference, "F3");
}

#[test]
fn op_f3dg_decomposition_correct() {
    let reference = pecos_core::unitary_rep::SZ(0).dg() * pecos_core::unitary_rep::SX(0);
    check_1q_clifford(op::F3dg(0), reference, "F3dg");
}

#[test]
fn op_f4_decomposition_correct() {
    let reference = pecos_core::unitary_rep::SZ(0) * pecos_core::unitary_rep::SX(0);
    check_1q_clifford(op::F4(0), reference, "F4");
}

#[test]
fn op_f4dg_decomposition_correct() {
    let reference = pecos_core::unitary_rep::SX(0).dg() * pecos_core::unitary_rep::SZ(0).dg();
    check_1q_clifford(op::F4dg(0), reference, "F4dg");
}

// ============================================================================
// 2-qubit Clifford gates
// ============================================================================

#[test]
fn op_cx_matches_unitary_rep() {
    check_2q_clifford(op::CX(0, 1), pecos_core::unitary_rep::CX(0, 1), "CX");
}

#[test]
fn op_cy_matches_unitary_rep() {
    check_2q_clifford(op::CY(0, 1), pecos_core::unitary_rep::CY(0, 1), "CY");
}

#[test]
fn op_cz_matches_unitary_rep() {
    check_2q_clifford(op::CZ(0, 1), pecos_core::unitary_rep::CZ(0, 1), "CZ");
}

#[test]
fn op_swap_matches_unitary_rep() {
    check_2q_clifford(op::SWAP(0, 1), pecos_core::unitary_rep::SWAP(0, 1), "SWAP");
}

#[test]
fn op_sxx_matches_rxx_quarter() {
    check_2q_clifford(
        op::SXX(0, 1),
        pecos_core::unitary_rep::RXX(Angle64::QUARTER_TURN, 0, 1),
        "SXX",
    );
}

#[test]
fn op_sxxdg_matches_rxx_three_quarters() {
    check_2q_clifford(
        op::SXXdg(0, 1),
        pecos_core::unitary_rep::RXX(Angle64::THREE_QUARTERS_TURN, 0, 1),
        "SXXdg",
    );
}

#[test]
fn op_syy_matches_ryy_quarter() {
    check_2q_clifford(
        op::SYY(0, 1),
        pecos_core::unitary_rep::RYY(Angle64::QUARTER_TURN, 0, 1),
        "SYY",
    );
}

#[test]
fn op_syydg_matches_ryy_three_quarters() {
    check_2q_clifford(
        op::SYYdg(0, 1),
        pecos_core::unitary_rep::RYY(Angle64::THREE_QUARTERS_TURN, 0, 1),
        "SYYdg",
    );
}

#[test]
fn op_szz_matches_unitary_rep() {
    check_2q_clifford(op::SZZ(0, 1), pecos_core::unitary_rep::SZZ(0, 1), "SZZ");
}

#[test]
fn op_szzdg_matches_unitary_rep() {
    check_2q_clifford(
        op::SZZdg(0, 1),
        pecos_core::unitary_rep::SZZ(0, 1).dg(),
        "SZZdg",
    );
}

// ============================================================================
// iSWAP and G gates — verify unitarity and dagger pairs
// ============================================================================

#[test]
fn op_iswap_is_unitary() {
    let mat = op::ISWAP(0, 1).into_unitary().unwrap().to_matrix();
    let n = mat.nrows();
    let product = mat.adjoint() * &mat;
    let identity = DMatrix::<Complex64>::identity(n, n);
    let diff = (&product - &identity).norm();
    assert!(diff < 1e-10, "iSWAP matrix is not unitary, diff = {diff}");
}

#[test]
fn op_iswap_dagger_pair() {
    let mat = op::ISWAP(0, 1).into_unitary().unwrap().to_matrix();
    let mat_dg = op::ISWAPdg(0, 1).into_unitary().unwrap().to_matrix();
    let product = &mat * &mat_dg;
    let n = product.nrows();
    let scale = product[(0, 0)];
    let scaled_id = &DMatrix::<Complex64>::identity(n, n) * scale;
    let diff = (&product - &scaled_id).norm();
    assert!(
        diff < 1e-10,
        "iSWAP * iSWAPdg is not identity (up to phase), diff = {diff}"
    );
}

#[test]
fn op_g_is_unitary() {
    let mat = op::G(0, 1).into_unitary().unwrap().to_matrix();
    let n = mat.nrows();
    let product = mat.adjoint() * &mat;
    let identity = DMatrix::<Complex64>::identity(n, n);
    let diff = (&product - &identity).norm();
    assert!(diff < 1e-10, "G matrix is not unitary, diff = {diff}");
}

#[test]
fn op_g_dagger_pair() {
    let mat = op::G(0, 1).into_unitary().unwrap().to_matrix();
    let mat_dg = op::Gdg(0, 1).into_unitary().unwrap().to_matrix();
    let product = &mat * &mat_dg;
    let n = product.nrows();
    let scale = product[(0, 0)];
    let scaled_id = &DMatrix::<Complex64>::identity(n, n) * scale;
    let diff = (&product - &scaled_id).norm();
    assert!(
        diff < 1e-10,
        "G * Gdg is not identity (up to phase), diff = {diff}"
    );
}

// ============================================================================
// Dagger pairs: gate * gate† = identity (via matrix)
// ============================================================================

#[test]
fn dagger_pairs_are_inverse() {
    let pairs: Vec<(&str, pecos_core::Op, pecos_core::Op)> = vec![
        ("H", op::H(0), op::H(0).dg()),
        ("SX", op::SX(0), op::SXdg(0)),
        ("SY", op::SY(0), op::SYdg(0)),
        ("SZ", op::SZ(0), op::SZdg(0)),
        ("F", op::F(0), op::Fdg(0)),
        ("F2", op::F2(0), op::F2dg(0)),
        ("F3", op::F3(0), op::F3dg(0)),
        ("F4", op::F4(0), op::F4dg(0)),
    ];

    let identity_2x2 = DMatrix::<Complex64>::identity(2, 2);

    for (name, gate, gate_dg) in pairs {
        let mat = gate.into_unitary().unwrap().to_matrix();
        let mat_dg = gate_dg.into_unitary().unwrap().to_matrix();
        let product = &mat * &mat_dg;

        // Check product is proportional to identity
        let scale = product[(0, 0)];
        let scaled_id = &identity_2x2 * scale;
        let diff = (&product - &scaled_id).norm();
        assert!(
            diff < 1e-10,
            "{name} * {name}dg is not identity (up to phase), diff = {diff}"
        );
    }
}
