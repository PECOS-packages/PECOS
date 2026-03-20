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

//! Tests for `UnitaryRep` <-> `CliffordRep` round-trips and `is_clifford` consistency.

use pecos_core::Angle64;
use pecos_core::clifford::Clifford;
use pecos_core::gate_type::GateType;
use pecos_core::unitary_rep::{RotationType, Unitary, UnitaryRep};

// ============================================================================
// is_clifford: GateType
// ============================================================================

#[test]
fn is_clifford_1q_named_gates() {
    let clifford_gates = [
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
    ];
    for gt in clifford_gates {
        let u = Unitary::Named(gt);
        assert!(u.is_clifford(), "Unitary::Named({gt:?}) should be Clifford");
    }
}

#[test]
fn is_clifford_2q_named_gates() {
    let clifford_gates = [
        GateType::CX,
        GateType::CY,
        GateType::CZ,
        GateType::SWAP,
        GateType::SXX,
        GateType::SXXdg,
        GateType::SYY,
        GateType::SYYdg,
        GateType::SZZ,
        GateType::SZZdg,
    ];
    for gt in clifford_gates {
        let u = Unitary::Named(gt);
        assert!(u.is_clifford(), "Unitary::Named({gt:?}) should be Clifford");
    }
}

#[test]
fn is_not_clifford_non_clifford_named_gates() {
    let non_clifford = [GateType::T, GateType::Tdg, GateType::CH, GateType::CCX];
    for gt in non_clifford {
        let u = Unitary::Named(gt);
        assert!(
            !u.is_clifford(),
            "Unitary::Named({gt:?}) should NOT be Clifford"
        );
    }
}

#[test]
fn is_clifford_quarter_turn_rotations() {
    for rot in [RotationType::RX, RotationType::RY, RotationType::RZ] {
        let u = Unitary::Rotation {
            rotation_type: rot,
            angle: Angle64::QUARTER_TURN,
        };
        assert!(
            u.is_clifford(),
            "{rot:?} at quarter turn should be Clifford"
        );
    }
}

#[test]
fn is_clifford_2q_quarter_turn_rotations() {
    for rot in [RotationType::RXX, RotationType::RYY, RotationType::RZZ] {
        let u = Unitary::Rotation {
            rotation_type: rot,
            angle: Angle64::QUARTER_TURN,
        };
        assert!(
            u.is_clifford(),
            "{rot:?} at quarter turn should be Clifford"
        );
    }
}

// ============================================================================
// to_clifford_rep round-trips via UnitaryRep
// ============================================================================

#[test]
fn to_clifford_rep_1q_named_gates() {
    let gates = [
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
    ];
    for gt in gates {
        let ur = Unitary::Named(gt).on_qubit(0);
        let cr = ur.to_clifford_rep(1);
        assert!(
            cr.is_some(),
            "to_clifford_rep should succeed for Named({gt:?})"
        );
    }
}

#[test]
fn to_clifford_rep_2q_named_gates() {
    let gates = [
        GateType::CX,
        GateType::CY,
        GateType::CZ,
        GateType::SWAP,
        GateType::SXX,
        GateType::SXXdg,
        GateType::SYY,
        GateType::SYYdg,
        GateType::SZZ,
        GateType::SZZdg,
    ];
    for gt in gates {
        let ur = Unitary::Named(gt).on_qubits(0, 1);
        let cr = ur.to_clifford_rep(2);
        assert!(
            cr.is_some(),
            "to_clifford_rep should succeed for Named({gt:?})"
        );
    }
}

#[test]
fn to_clifford_rep_1q_rotations_at_quarter_turn() {
    for rot in [RotationType::RX, RotationType::RY, RotationType::RZ] {
        let ur = UnitaryRep::Gate(
            Unitary::Rotation {
                rotation_type: rot,
                angle: Angle64::QUARTER_TURN,
            },
            smallvec::smallvec![0],
        );
        let cr = ur.to_clifford_rep(1);
        assert!(
            cr.is_some(),
            "to_clifford_rep should succeed for {rot:?} at quarter turn"
        );
    }
}

#[test]
fn to_clifford_rep_2q_rotations_at_quarter_turn() {
    for rot in [RotationType::RXX, RotationType::RYY, RotationType::RZZ] {
        let ur = UnitaryRep::Gate(
            Unitary::Rotation {
                rotation_type: rot,
                angle: Angle64::QUARTER_TURN,
            },
            smallvec::smallvec![0, 1],
        );
        let cr = ur.to_clifford_rep(2);
        assert!(
            cr.is_some(),
            "to_clifford_rep should succeed for {rot:?} at quarter turn"
        );
    }
}

#[test]
fn to_clifford_rep_2q_rotations_at_three_quarter_turn() {
    for rot in [RotationType::RXX, RotationType::RYY, RotationType::RZZ] {
        let ur = UnitaryRep::Gate(
            Unitary::Rotation {
                rotation_type: rot,
                angle: Angle64::THREE_QUARTERS_TURN,
            },
            smallvec::smallvec![0, 1],
        );
        let cr = ur.to_clifford_rep(2);
        assert!(
            cr.is_some(),
            "to_clifford_rep should succeed for {rot:?} at three-quarter turn"
        );
    }
}

#[test]
fn to_clifford_rep_2q_rotations_at_half_turn() {
    for rot in [RotationType::RXX, RotationType::RYY, RotationType::RZZ] {
        let ur = UnitaryRep::Gate(
            Unitary::Rotation {
                rotation_type: rot,
                angle: Angle64::HALF_TURN,
            },
            smallvec::smallvec![0, 1],
        );
        let cr = ur.to_clifford_rep(2);
        assert!(
            cr.is_some(),
            "to_clifford_rep should succeed for {rot:?} at half turn"
        );
    }
}

// ============================================================================
// Verify CliffordRep from to_clifford_rep matches the Clifford enum's version
// ============================================================================

#[test]
fn to_clifford_rep_succeeds_for_all_1q_clifford_decompositions() {
    // Verify that to_clifford_rep succeeds for all 1q Clifford UnitaryRep decompositions.
    // Note: decompositions like SY*Z for H2 may produce a CliffordRep that differs from
    // Clifford::H2.on_qubit(0) by sign (global phase), so we only check success here.
    for &cliff in Clifford::all_1q() {
        let ur = cliff.to_unitary_rep_on_qubit(0usize);
        let cr = ur.to_clifford_rep(1);
        assert!(
            cr.is_some(),
            "to_clifford_rep failed for Clifford::{cliff:?}"
        );
    }
}

#[test]
fn to_clifford_rep_exact_match_for_named_1q_gates() {
    // For gates with a direct Named GateType, the round-trip should be exact.
    let gates = [
        (Clifford::I, GateType::I),
        (Clifford::X, GateType::X),
        (Clifford::Y, GateType::Y),
        (Clifford::Z, GateType::Z),
        (Clifford::H, GateType::H),
        (Clifford::SX, GateType::SX),
        (Clifford::SXdg, GateType::SXdg),
        (Clifford::SY, GateType::SY),
        (Clifford::SYdg, GateType::SYdg),
        (Clifford::SZ, GateType::SZ),
        (Clifford::SZdg, GateType::SZdg),
        (Clifford::F, GateType::F),
        (Clifford::Fdg, GateType::Fdg),
    ];
    for (cliff, gt) in gates {
        let ur = Unitary::Named(gt).on_qubit(0);
        let cr = ur
            .to_clifford_rep(1)
            .unwrap_or_else(|| panic!("to_clifford_rep failed for Named({gt:?})"));
        let expected = cliff.on_qubit(0);
        assert_eq!(
            cr, expected,
            "CliffordRep mismatch for {cliff:?} via Named({gt:?})"
        );
    }
}

#[test]
fn to_clifford_rep_matches_clifford_on_qubits_standard_2q() {
    // Test the 2q gates that have GateType entries and can be tested via Named path
    let gates = [
        (Clifford::CX, GateType::CX),
        (Clifford::CY, GateType::CY),
        (Clifford::CZ, GateType::CZ),
        (Clifford::SWAP, GateType::SWAP),
        (Clifford::SXX, GateType::SXX),
        (Clifford::SXXdg, GateType::SXXdg),
        (Clifford::SYY, GateType::SYY),
        (Clifford::SYYdg, GateType::SYYdg),
        (Clifford::SZZ, GateType::SZZ),
        (Clifford::SZZdg, GateType::SZZdg),
    ];
    for (cliff, gt) in gates {
        let ur = Unitary::Named(gt).on_qubits(0, 1);
        let cr_from_ur = ur.to_clifford_rep(2);
        assert!(
            cr_from_ur.is_some(),
            "to_clifford_rep failed for {cliff:?} via Named({gt:?})"
        );
        let cr_direct = cliff.on_qubits(0, 1);
        assert_eq!(
            cr_from_ur.unwrap(),
            cr_direct,
            "CliffordRep mismatch for {cliff:?}"
        );
    }
}

// ============================================================================
// rotation_to_gate_type: 2q rotations
// ============================================================================

#[test]
fn rotation_to_gate_type_rxx_quarter_and_three_quarter() {
    use pecos_core::unitary_rep::rotation_to_gate_type;
    assert_eq!(
        rotation_to_gate_type(RotationType::RXX, Angle64::QUARTER_TURN),
        Some(GateType::SXX)
    );
    assert_eq!(
        rotation_to_gate_type(RotationType::RXX, Angle64::THREE_QUARTERS_TURN),
        Some(GateType::SXXdg)
    );
    assert_eq!(
        rotation_to_gate_type(RotationType::RXX, Angle64::HALF_TURN),
        None,
        "RXX at half turn has no named gate"
    );
}

#[test]
fn rotation_to_gate_type_ryy_quarter_and_three_quarter() {
    use pecos_core::unitary_rep::rotation_to_gate_type;
    assert_eq!(
        rotation_to_gate_type(RotationType::RYY, Angle64::QUARTER_TURN),
        Some(GateType::SYY)
    );
    assert_eq!(
        rotation_to_gate_type(RotationType::RYY, Angle64::THREE_QUARTERS_TURN),
        Some(GateType::SYYdg)
    );
    assert_eq!(
        rotation_to_gate_type(RotationType::RYY, Angle64::HALF_TURN),
        None,
        "RYY at half turn has no named gate"
    );
}

#[test]
fn rotation_to_gate_type_rzz_quarter_and_three_quarter() {
    use pecos_core::unitary_rep::rotation_to_gate_type;
    assert_eq!(
        rotation_to_gate_type(RotationType::RZZ, Angle64::QUARTER_TURN),
        Some(GateType::SZZ)
    );
    assert_eq!(
        rotation_to_gate_type(RotationType::RZZ, Angle64::THREE_QUARTERS_TURN),
        Some(GateType::SZZdg)
    );
}

// ============================================================================
// is_self_adjoint (tested via dg() behavior on UnitaryRep)
// ============================================================================

#[test]
fn dg_is_identity_for_self_adjoint_gates() {
    // Self-adjoint 1q gates: dg() returns a clone (not Adjoint wrapper)
    let self_adjoint_1q = [
        GateType::I,
        GateType::X,
        GateType::Y,
        GateType::Z,
        GateType::H,
    ];
    for gt in self_adjoint_1q {
        let ur = Unitary::Named(gt).on_qubit(0);
        let dg = ur.dg();
        assert_eq!(
            ur, dg,
            "dg() of self-adjoint Named({gt:?}) should equal itself"
        );
    }

    // Self-adjoint 2q gates
    let self_adjoint_2q = [GateType::CX, GateType::CY, GateType::CZ, GateType::SWAP];
    for gt in self_adjoint_2q {
        let ur = Unitary::Named(gt).on_qubits(0, 1);
        let dg = ur.dg();
        assert_eq!(
            ur, dg,
            "dg() of self-adjoint Named({gt:?}) should equal itself"
        );
    }

    // CCX (3-qubit, self-adjoint)
    let ur_ccx = UnitaryRep::Gate(Unitary::Named(GateType::CCX), smallvec::smallvec![0, 1, 2]);
    let dg_ccx = ur_ccx.dg();
    assert_eq!(
        ur_ccx, dg_ccx,
        "dg() of self-adjoint CCX should equal itself"
    );
}

#[test]
fn dg_wraps_adjoint_for_non_self_adjoint_gates() {
    // Non-self-adjoint Named gates: dg() wraps in Adjoint
    let not_self_adjoint = [
        GateType::SX,
        GateType::SXdg,
        GateType::SY,
        GateType::SYdg,
        GateType::SZ,
        GateType::SZdg,
        GateType::T,
        GateType::Tdg,
        GateType::F,
        GateType::Fdg,
    ];
    for gt in not_self_adjoint {
        let ur = Unitary::Named(gt).on_qubit(0);
        let dg = ur.dg();
        assert_ne!(
            ur, dg,
            "dg() of non-self-adjoint Named({gt:?}) should differ from itself"
        );
    }
}

// ============================================================================
// rotation_to_gate_type: edge cases
// ============================================================================

#[test]
fn rotation_to_gate_type_half_turn_maps_to_paulis() {
    use pecos_core::unitary_rep::rotation_to_gate_type;
    assert_eq!(
        rotation_to_gate_type(RotationType::RX, Angle64::HALF_TURN),
        Some(GateType::X)
    );
    assert_eq!(
        rotation_to_gate_type(RotationType::RY, Angle64::HALF_TURN),
        Some(GateType::Y)
    );
    assert_eq!(
        rotation_to_gate_type(RotationType::RZ, Angle64::HALF_TURN),
        Some(GateType::Z)
    );
}

#[test]
fn rotation_to_gate_type_zero_and_full_turn_return_none() {
    use pecos_core::unitary_rep::rotation_to_gate_type;
    for rot in [RotationType::RX, RotationType::RY, RotationType::RZ] {
        assert_eq!(
            rotation_to_gate_type(rot, Angle64::ZERO),
            None,
            "{rot:?} at zero should have no named gate"
        );
        assert_eq!(
            rotation_to_gate_type(rot, Angle64::FULL_TURN),
            None,
            "{rot:?} at full turn should have no named gate"
        );
    }
}

#[test]
fn rotation_to_gate_type_eighth_turn_maps_to_t() {
    use pecos_core::unitary_rep::rotation_to_gate_type;
    let eighth = Angle64::HALF_TURN / 4u64; // pi/4
    assert_eq!(
        rotation_to_gate_type(RotationType::RZ, eighth),
        Some(GateType::T)
    );
    // RX and RY at pi/4 have no named gate
    assert_eq!(rotation_to_gate_type(RotationType::RX, eighth), None);
    assert_eq!(rotation_to_gate_type(RotationType::RY, eighth), None);
}

// ============================================================================
// dg() involution: dg(dg(x)) == x
// ============================================================================

#[test]
fn dg_involution_rotation() {
    for rot in [RotationType::RX, RotationType::RY, RotationType::RZ] {
        for angle in [
            Angle64::QUARTER_TURN,
            Angle64::HALF_TURN,
            Angle64::THREE_QUARTERS_TURN,
        ] {
            let ur = UnitaryRep::Gate(
                Unitary::Rotation {
                    rotation_type: rot,
                    angle,
                },
                smallvec::smallvec![0],
            );
            assert_eq!(
                ur.dg().dg(),
                ur,
                "dg(dg({rot:?}({angle:?}))) should equal original"
            );
        }
    }
}

#[test]
fn dg_involution_compose() {
    let composed = pecos_core::unitary_rep::H(0) * pecos_core::unitary_rep::SZ(0);
    assert_eq!(
        composed.dg().dg(),
        composed,
        "dg(dg(H*SZ)) should equal original"
    );
}

#[test]
fn dg_involution_tensor() {
    let tensor = pecos_core::unitary_rep::H(0) & pecos_core::unitary_rep::X(1);
    assert_eq!(
        tensor.dg().dg(),
        tensor,
        "dg(dg(H&X)) should equal original"
    );
}

#[test]
fn dg_involution_phase() {
    let phased =
        pecos_core::unitary_rep::phase(Angle64::QUARTER_TURN) * pecos_core::unitary_rep::X(0);
    assert_eq!(
        phased.dg().dg(),
        phased,
        "dg(dg(phase*X)) should equal original"
    );
}

#[test]
fn dg_involution_adjoint() {
    // Adjoint(op).dg() should unwrap back to op
    let ur = Unitary::Named(GateType::SX).on_qubit(0);
    let adj = ur.dg(); // wraps in Adjoint
    assert_eq!(adj.dg(), ur, "dg(Adjoint(SX)) should unwrap to SX");
}

// ============================================================================
// dg() negates rotation angle
// ============================================================================

#[test]
fn dg_negates_rotation_angle() {
    let angle = Angle64::QUARTER_TURN;
    let neg_angle = Angle64::THREE_QUARTERS_TURN; // == -pi/2 mod 2pi
    for rot in [RotationType::RX, RotationType::RY, RotationType::RZ] {
        let ur = UnitaryRep::Gate(
            Unitary::Rotation {
                rotation_type: rot,
                angle,
            },
            smallvec::smallvec![0],
        );
        let expected = UnitaryRep::Gate(
            Unitary::Rotation {
                rotation_type: rot,
                angle: neg_angle,
            },
            smallvec::smallvec![0],
        );
        assert_eq!(
            ur.dg(),
            expected,
            "dg({rot:?}(quarter)) should be {rot:?}(three_quarter)"
        );
    }
}

// ============================================================================
// is_clifford on composite UnitaryRep
// ============================================================================

#[test]
fn is_clifford_composed_cliffords() {
    let composed = pecos_core::unitary_rep::H(0) * pecos_core::unitary_rep::SZ(0);
    assert!(composed.is_clifford(), "H * SZ should be Clifford");
}

#[test]
fn is_clifford_tensor_cliffords() {
    let tensor = pecos_core::unitary_rep::H(0) & pecos_core::unitary_rep::X(1);
    assert!(tensor.is_clifford(), "H & X should be Clifford");
}

#[test]
fn is_not_clifford_composed_with_non_clifford() {
    let t_gate = UnitaryRep::Gate(Unitary::Named(GateType::T), smallvec::smallvec![0]);
    let composed = pecos_core::unitary_rep::H(0) * t_gate;
    assert!(!composed.is_clifford(), "H * T should NOT be Clifford");
}

#[test]
fn is_not_clifford_tensor_with_non_clifford() {
    let t_gate = UnitaryRep::Gate(Unitary::Named(GateType::T), smallvec::smallvec![1]);
    let tensor = pecos_core::unitary_rep::H(0) & t_gate;
    assert!(!tensor.is_clifford(), "H & T should NOT be Clifford");
}

#[test]
fn is_clifford_adjoint_of_clifford() {
    let adj = pecos_core::unitary_rep::SX(0).dg();
    assert!(adj.is_clifford(), "SX.dg() should be Clifford");
}

#[test]
fn is_not_clifford_adjoint_of_non_clifford() {
    let t_gate = UnitaryRep::Gate(Unitary::Named(GateType::T), smallvec::smallvec![0]);
    assert!(!t_gate.dg().is_clifford(), "T.dg() should NOT be Clifford");
}

#[test]
fn is_clifford_phased_clifford() {
    let phased =
        pecos_core::unitary_rep::phase(Angle64::QUARTER_TURN) * pecos_core::unitary_rep::H(0);
    assert!(phased.is_clifford(), "phase(pi/2) * H should be Clifford");
}

// ============================================================================
// to_clifford_rep: zero-angle rotation is identity
// ============================================================================

#[test]
fn to_clifford_rep_zero_angle_is_identity() {
    use pecos_core::clifford_rep::CliffordRep;
    for rot in [RotationType::RX, RotationType::RY, RotationType::RZ] {
        let ur = UnitaryRep::Gate(
            Unitary::Rotation {
                rotation_type: rot,
                angle: Angle64::ZERO,
            },
            smallvec::smallvec![0],
        );
        let cr = ur.to_clifford_rep(1);
        assert!(
            cr.is_some(),
            "to_clifford_rep should succeed for {rot:?} at zero"
        );
        assert_eq!(
            cr.unwrap(),
            CliffordRep::identity(1),
            "{rot:?}(0) should produce identity CliffordRep"
        );
    }
}

#[test]
fn to_clifford_rep_full_turn_is_identity() {
    use pecos_core::clifford_rep::CliffordRep;
    for rot in [RotationType::RX, RotationType::RY, RotationType::RZ] {
        let ur = UnitaryRep::Gate(
            Unitary::Rotation {
                rotation_type: rot,
                angle: Angle64::FULL_TURN,
            },
            smallvec::smallvec![0],
        );
        let cr = ur.to_clifford_rep(1);
        assert!(
            cr.is_some(),
            "to_clifford_rep should succeed for {rot:?} at full turn"
        );
        assert_eq!(
            cr.unwrap(),
            CliffordRep::identity(1),
            "{rot:?}(2pi) should produce identity CliffordRep"
        );
    }
}

// ============================================================================
// Verify SZZ vs SZZdg are actually different
// ============================================================================

#[test]
fn szz_and_szzdg_are_distinct() {
    use pecos_core::clifford_rep::CliffordRep;
    let szz = CliffordRep::szz(0, 1);
    let szzdg = CliffordRep::szzdg(0, 1);
    assert_ne!(szz, szzdg, "SZZ and SZZdg CliffordReps should differ");

    // Also verify through the rotation path
    let ur_szz = UnitaryRep::Gate(
        Unitary::Rotation {
            rotation_type: RotationType::RZZ,
            angle: Angle64::QUARTER_TURN,
        },
        smallvec::smallvec![0, 1],
    );
    let ur_szzdg = UnitaryRep::Gate(
        Unitary::Rotation {
            rotation_type: RotationType::RZZ,
            angle: Angle64::THREE_QUARTERS_TURN,
        },
        smallvec::smallvec![0, 1],
    );
    let cr_szz = ur_szz.to_clifford_rep(2).unwrap();
    let cr_szzdg = ur_szzdg.to_clifford_rep(2).unwrap();
    assert_ne!(
        cr_szz, cr_szzdg,
        "RZZ(pi/2) and RZZ(3pi/2) should produce different CliffordReps"
    );
}
