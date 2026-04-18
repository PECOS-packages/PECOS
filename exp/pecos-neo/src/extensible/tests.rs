//! Tests for the extensible gate system.
#![allow(clippy::float_cmp)]

use super::validator::GateForValidation;
use super::*;
use pecos_core::{Angle64, QubitId};

// ============================================================================
// GateId Tests
// ============================================================================

#[test]
fn test_gate_id_size() {
    // GateId must be compact - exactly 2 bytes
    assert_eq!(std::mem::size_of::<GateId>(), 2);
}

#[test]
fn test_gate_id_core_range() {
    // IDs 0-255 are core gates
    assert!(GateId(0).is_core());
    assert!(GateId(255).is_core());
    assert!(!GateId(256).is_core());
}

#[test]
fn test_gate_id_user_range() {
    // IDs 256+ are user-defined
    assert!(!GateId(0).is_user_defined());
    assert!(!GateId(255).is_user_defined());
    assert!(GateId(256).is_user_defined());
    assert!(GateId(1000).is_user_defined());
}

#[test]
fn test_gate_id_into_usize() {
    // GateId should convert to usize for indexing
    let id = GateId(42);
    let idx: usize = id.into();
    assert_eq!(idx, 42);
}

#[test]
fn test_gate_id_equality() {
    assert_eq!(GateId(10), GateId(10));
    assert_ne!(GateId(10), GateId(11));
}

#[test]
fn test_gate_id_ordering() {
    assert!(GateId(5) < GateId(10));
    assert!(GateId(256) > GateId(255));
}

#[test]
fn test_gate_id_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(GateId(1));
    set.insert(GateId(2));
    set.insert(GateId(1)); // Duplicate
    assert_eq!(set.len(), 2);
}

// ============================================================================
// Gate Constants Tests
// ============================================================================

#[test]
fn test_core_gate_constants_are_core() {
    // All gate constants should be in core range
    assert!(gates::I.is_core());
    assert!(gates::X.is_core());
    assert!(gates::Y.is_core());
    assert!(gates::Z.is_core());
    assert!(gates::H.is_core());
    assert!(gates::CX.is_core());
    assert!(gates::CZ.is_core());
    assert!(gates::RZ.is_core());
    assert!(gates::MZ.is_core());
    assert!(gates::PZ.is_core());
}

#[test]
fn test_core_gate_constants_unique() {
    // All gate constants should have unique IDs
    let ids = [
        gates::I,
        gates::X,
        gates::Y,
        gates::Z,
        gates::H,
        gates::SX,
        gates::SY,
        gates::SZ,
        gates::T,
        gates::CX,
        gates::CZ,
        gates::RZ,
        gates::RX,
        gates::RY,
        gates::MZ,
        gates::PZ,
    ];

    for i in 0..ids.len() {
        for j in (i + 1)..ids.len() {
            assert_ne!(
                ids[i], ids[j],
                "Gate constants {:?} and {:?} have same ID",
                ids[i], ids[j]
            );
        }
    }
}

// ============================================================================
// GateSpec Tests
// ============================================================================

#[test]
fn test_gate_spec_creation() {
    let spec = GateSpec {
        name: "MyGate",
        quantum_arity: 2,
        angle_arity: 3,
        param_arity: 0,
        returns_result: false,
        category: GateCategory::TwoQubitUnitary,
    };

    assert_eq!(spec.name, "MyGate");
    assert_eq!(spec.quantum_arity, 2);
    assert_eq!(spec.angle_arity, 3);
    assert_eq!(spec.param_arity, 0);
    assert!(!spec.returns_result);
    assert_eq!(spec.category, GateCategory::TwoQubitUnitary);
}

#[test]
fn test_gate_spec_default() {
    let spec = GateSpec::default();

    assert_eq!(spec.name, "");
    assert_eq!(spec.quantum_arity, 1);
    assert_eq!(spec.angle_arity, 0);
    assert!(!spec.returns_result);
}

#[test]
fn test_gate_category_equality() {
    assert_eq!(
        GateCategory::SingleQubitUnitary,
        GateCategory::SingleQubitUnitary
    );
    assert_ne!(
        GateCategory::SingleQubitUnitary,
        GateCategory::TwoQubitUnitary
    );
    assert_ne!(GateCategory::Custom(1), GateCategory::Custom(2));
    assert_eq!(GateCategory::Custom(5), GateCategory::Custom(5));
}

// ============================================================================
// GateRegistry Tests
// ============================================================================

#[test]
fn test_registry_new_has_core_gates() {
    let registry = GateRegistry::new();

    // Core gates should be pre-registered
    assert!(registry.get(gates::X).is_some());
    assert!(registry.get(gates::H).is_some());
    assert!(registry.get(gates::CX).is_some());
    assert!(registry.get(gates::RZ).is_some());
    assert!(registry.get(gates::MZ).is_some());
    assert!(registry.get(gates::PZ).is_some());
}

#[test]
fn test_registry_core_gate_specs_correct() {
    let registry = GateRegistry::new();

    // H gate
    let h = registry.get(gates::H).unwrap();
    assert_eq!(h.name, "H");
    assert_eq!(h.quantum_arity, 1);
    assert_eq!(h.angle_arity, 0);
    assert!(!h.returns_result);

    // CX gate
    let cx = registry.get(gates::CX).unwrap();
    assert_eq!(cx.name, "CX");
    assert_eq!(cx.quantum_arity, 2);
    assert_eq!(cx.angle_arity, 0);

    // RZ gate
    let rz = registry.get(gates::RZ).unwrap();
    assert_eq!(rz.name, "RZ");
    assert_eq!(rz.quantum_arity, 1);
    assert_eq!(rz.angle_arity, 1);

    // Measure gate
    let meas = registry.get(gates::MZ).unwrap();
    assert_eq!(meas.name, "MZ");
    assert!(meas.returns_result);
}

#[test]
fn test_registry_register_user_gate() {
    let mut registry = GateRegistry::new();

    let id = registry.register(GateSpec {
        name: "MyRotation",
        quantum_arity: 2,
        angle_arity: 3,
        param_arity: 0,
        returns_result: false,
        category: GateCategory::TwoQubitUnitary,
    });

    assert!(id.is_user_defined());
    assert_eq!(id.0, 256); // First user gate

    let spec = registry.get(id).unwrap();
    assert_eq!(spec.name, "MyRotation");
    assert_eq!(spec.quantum_arity, 2);
    assert_eq!(spec.angle_arity, 3);
}

#[test]
fn test_registry_register_multiple_user_gates() {
    let mut registry = GateRegistry::new();

    let id1 = registry.register(GateSpec {
        name: "Gate1",
        ..Default::default()
    });
    let id2 = registry.register(GateSpec {
        name: "Gate2",
        ..Default::default()
    });
    let id3 = registry.register(GateSpec {
        name: "Gate3",
        ..Default::default()
    });

    assert_eq!(id1.0, 256);
    assert_eq!(id2.0, 257);
    assert_eq!(id3.0, 258);

    assert_eq!(registry.get(id1).unwrap().name, "Gate1");
    assert_eq!(registry.get(id2).unwrap().name, "Gate2");
    assert_eq!(registry.get(id3).unwrap().name, "Gate3");
}

#[test]
fn test_registry_lookup_core_gate_by_name() {
    let registry = GateRegistry::new();

    assert_eq!(registry.lookup("H"), Some(gates::H));
    assert_eq!(registry.lookup("CX"), Some(gates::CX));
    assert_eq!(registry.lookup("RZ"), Some(gates::RZ));
    assert_eq!(registry.lookup("MZ"), Some(gates::MZ));
}

#[test]
fn test_registry_lookup_user_gate_by_name() {
    let mut registry = GateRegistry::new();

    let id = registry.register(GateSpec {
        name: "MyGate",
        ..Default::default()
    });

    assert_eq!(registry.lookup("MyGate"), Some(id));
}

#[test]
fn test_registry_lookup_nonexistent() {
    let registry = GateRegistry::new();

    assert_eq!(registry.lookup("NonExistentGate"), None);
}

#[test]
fn test_registry_contains() {
    let mut registry = GateRegistry::new();

    assert!(registry.contains(gates::H));
    assert!(registry.contains(gates::CX));

    // Unregistered user gate
    assert!(!registry.contains(GateId(256)));

    // Register and check again
    let id = registry.register(GateSpec {
        name: "Test",
        ..Default::default()
    });
    assert!(registry.contains(id));
}

#[test]
fn test_registries_are_independent() {
    let mut registry1 = GateRegistry::new();
    let mut registry2 = GateRegistry::new();

    let id1 = registry1.register(GateSpec {
        name: "GateA",
        ..Default::default()
    });
    let id2 = registry2.register(GateSpec {
        name: "GateB",
        ..Default::default()
    });

    // Same ID value
    assert_eq!(id1.0, id2.0);

    // But different specs in each registry
    assert_eq!(registry1.get(id1).unwrap().name, "GateA");
    assert_eq!(registry2.get(id2).unwrap().name, "GateB");

    // Registry1 doesn't know about GateB
    assert!(registry1.lookup("GateB").is_none());
    assert!(registry2.lookup("GateA").is_none());
}

// ============================================================================
// GateSupportSet Tests
// ============================================================================

#[test]
fn test_support_set_empty() {
    let set = GateSupportSet::new();

    assert!(!set.contains(gates::H));
    assert!(!set.contains(gates::CX));
    assert!(!set.contains(GateId(256)));
}

#[test]
fn test_support_set_insert_contains() {
    let mut set = GateSupportSet::new();

    set.insert(gates::H);
    set.insert(gates::CX);

    assert!(set.contains(gates::H));
    assert!(set.contains(gates::CX));
    assert!(!set.contains(gates::RZ));
}

#[test]
fn test_support_set_user_gates() {
    let mut set = GateSupportSet::new();

    let user_gate = GateId(300);
    set.insert(user_gate);

    assert!(set.contains(user_gate));
    assert!(!set.contains(GateId(301)));
}

#[test]
fn test_support_set_iter() {
    let mut set = GateSupportSet::new();

    set.insert(gates::H);
    set.insert(gates::CX);
    set.insert(gates::RZ);

    let ids: Vec<GateId> = set.iter().collect();

    assert!(ids.contains(&gates::H));
    assert!(ids.contains(&gates::CX));
    assert!(ids.contains(&gates::RZ));
    assert_eq!(ids.len(), 3);
}

#[test]
fn test_support_set_union() {
    let mut set1 = GateSupportSet::new();
    set1.insert(gates::H);
    set1.insert(gates::X);

    let mut set2 = GateSupportSet::new();
    set2.insert(gates::CX);
    set2.insert(gates::X); // Overlap

    set1.union_with(&set2);

    assert!(set1.contains(gates::H));
    assert!(set1.contains(gates::X));
    assert!(set1.contains(gates::CX));
}

#[test]
fn test_support_set_difference() {
    let mut required = GateSupportSet::new();
    required.insert(gates::H);
    required.insert(gates::CX);
    required.insert(gates::RZ);

    let mut supported = GateSupportSet::new();
    supported.insert(gates::H);
    supported.insert(gates::CX);
    // RZ not supported

    let unsupported = required.difference(&supported);

    assert!(!unsupported.contains(gates::H));
    assert!(!unsupported.contains(gates::CX));
    assert!(unsupported.contains(gates::RZ));
}

// ============================================================================
// GateCanonicalizer Tests
// ============================================================================

#[test]
fn test_canonicalize_rz_zero() {
    let canon = GateCanonicalizer::standard();

    // RZ(0) = I
    let result = canon.canonicalize(gates::RZ, &[Angle64::ZERO]);
    assert_eq!(result, Some(gates::I));
}

#[test]
fn test_canonicalize_rz_quarter_turn() {
    let canon = GateCanonicalizer::standard();

    // RZ(π/2) = SZ
    let result = canon.canonicalize(gates::RZ, &[Angle64::QUARTER_TURN]);
    assert_eq!(result, Some(gates::SZ));
}

#[test]
fn test_canonicalize_rz_half_turn() {
    let canon = GateCanonicalizer::standard();

    // RZ(π) = Z
    let result = canon.canonicalize(gates::RZ, &[Angle64::HALF_TURN]);
    assert_eq!(result, Some(gates::Z));
}

#[test]
fn test_canonicalize_rz_t_gate() {
    let canon = GateCanonicalizer::standard();

    // RZ(π/4) = T
    let t_angle = Angle64::HALF_TURN / 4;
    let result = canon.canonicalize(gates::RZ, &[t_angle]);
    assert_eq!(result, Some(gates::T));
}

#[test]
fn test_canonicalize_rz_negative_quarter() {
    let canon = GateCanonicalizer::standard();

    // RZ(-π/2) = SZdg
    let neg_quarter = Angle64::ZERO - Angle64::QUARTER_TURN;
    let result = canon.canonicalize(gates::RZ, &[neg_quarter]);
    assert_eq!(result, Some(gates::SZdg));
}

#[test]
fn test_canonicalize_arbitrary_angle_returns_none() {
    let canon = GateCanonicalizer::standard();

    // RZ(0.123 turns) has no canonical form
    let arbitrary = Angle64::from_turns(0.123);
    let result = canon.canonicalize(gates::RZ, &[arbitrary]);
    assert_eq!(result, None);
}

#[test]
fn test_canonicalize_rx_half_turn() {
    let canon = GateCanonicalizer::standard();

    // RX(π) = X
    let result = canon.canonicalize(gates::RX, &[Angle64::HALF_TURN]);
    assert_eq!(result, Some(gates::X));
}

#[test]
fn test_canonicalize_rx_quarter_turn() {
    let canon = GateCanonicalizer::standard();

    // RX(π/2) = SX
    let result = canon.canonicalize(gates::RX, &[Angle64::QUARTER_TURN]);
    assert_eq!(result, Some(gates::SX));
}

#[test]
fn test_canonicalize_multi_angle_gate_returns_none() {
    let canon = GateCanonicalizer::standard();

    // Gates with multiple angles don't canonicalize (for now)
    let result = canon.canonicalize(gates::RZ, &[Angle64::QUARTER_TURN, Angle64::HALF_TURN]);
    assert_eq!(result, None);
}

#[test]
fn test_canonicalize_non_parameterized_gate_returns_none() {
    let canon = GateCanonicalizer::standard();

    // H gate has no angles, nothing to canonicalize
    let result = canon.canonicalize(gates::H, &[]);
    assert_eq!(result, None);
}

// ============================================================================
// Angle64 Exactness Tests (verifying our assumptions)
// ============================================================================

#[test]
fn test_angle64_quarter_turn_exact() {
    // Verify that 0.25 turns converts exactly to QUARTER_TURN
    let from_turns = Angle64::from_turns(0.25);
    assert_eq!(from_turns, Angle64::QUARTER_TURN);
}

#[test]
fn test_angle64_half_turn_exact() {
    let from_turns = Angle64::from_turns(0.5);
    assert_eq!(from_turns, Angle64::HALF_TURN);
}

#[test]
fn test_angle64_eighth_turn_exact() {
    // T gate angle: π/4 = 1/8 turn
    let from_turns = Angle64::from_turns(0.125);
    let expected = Angle64::HALF_TURN / 4;
    assert_eq!(from_turns, expected);
}

#[test]
fn test_angle64_arithmetic_preserves_exactness() {
    // Adding exact angles gives exact result
    let t = Angle64::HALF_TURN / 4; // π/4
    let s = t + t; // π/2

    assert_eq!(s, Angle64::QUARTER_TURN);
}

#[test]
fn test_angle64_negative_via_subtraction() {
    // -π/2 = 0 - π/2 = 3π/2 (wraps around)
    let neg_quarter = Angle64::ZERO - Angle64::QUARTER_TURN;
    assert_eq!(neg_quarter, Angle64::THREE_QUARTERS_TURN);
}

// ============================================================================
// Additional Edge Case Tests
// ============================================================================

#[test]
fn test_gate_id_max_core() {
    // Edge of core range
    let max_core = GateId(255);
    let first_user = GateId(256);

    assert!(max_core.is_core());
    assert!(!max_core.is_user_defined());
    assert!(!first_user.is_core());
    assert!(first_user.is_user_defined());
}

#[test]
fn test_gate_id_max_value() {
    // Maximum possible ID
    let max_id = GateId(u16::MAX);
    assert!(max_id.is_user_defined());
    assert_eq!(usize::from(max_id), 65535);
}

#[test]
fn test_registry_get_unregistered_core() {
    let registry = GateRegistry::new();

    // Some core IDs might not be registered
    // (gaps in the ID space are allowed)
    let unused_id = GateId(200); // Likely unused
    assert!(registry.get(unused_id).is_none());
}

#[test]
fn test_registry_get_unregistered_user() {
    let registry = GateRegistry::new();

    // User gates not registered should return None
    assert!(registry.get(GateId(256)).is_none());
    assert!(registry.get(GateId(1000)).is_none());
}

#[test]
fn test_registry_iter_ids() {
    let registry = GateRegistry::new();

    let ids: Vec<GateId> = registry.iter_ids().collect();

    // Should have many core gates
    assert!(ids.len() > 20);

    // All core gates should be present
    assert!(ids.contains(&gates::H));
    assert!(ids.contains(&gates::CX));
    assert!(ids.contains(&gates::MZ));
}

#[test]
fn test_registry_iter() {
    let registry = GateRegistry::new();

    let specs: Vec<(GateId, &GateSpec)> = registry.iter().collect();

    // Find H gate
    let h = specs.iter().find(|(id, _)| *id == gates::H);
    assert!(h.is_some());
    assert_eq!(h.unwrap().1.name, "H");
}

#[test]
fn test_registry_user_gate_count() {
    let mut registry = GateRegistry::new();

    assert_eq!(registry.user_gate_count(), 0);

    registry.register(GateSpec {
        name: "A",
        ..Default::default()
    });
    assert_eq!(registry.user_gate_count(), 1);

    registry.register(GateSpec {
        name: "B",
        ..Default::default()
    });
    assert_eq!(registry.user_gate_count(), 2);
}

#[test]
fn test_support_set_large_id() {
    let mut set = GateSupportSet::new();

    // Insert a large user gate ID
    let large_id = GateId(10000);
    set.insert(large_id);

    assert!(set.contains(large_id));
    assert!(!set.contains(GateId(9999)));
    assert!(!set.contains(GateId(10001)));
}

#[test]
fn test_support_set_is_subset() {
    let mut small = GateSupportSet::new();
    small.insert(gates::H);
    small.insert(gates::X);

    let mut large = GateSupportSet::new();
    large.insert(gates::H);
    large.insert(gates::X);
    large.insert(gates::CX);

    assert!(small.is_subset_of(&large));
    assert!(!large.is_subset_of(&small));
}

#[test]
fn test_support_set_empty_operations() {
    let empty = GateSupportSet::new();
    let mut set = GateSupportSet::new();
    set.insert(gates::H);

    // Union with empty
    set.union_with(&empty);
    assert_eq!(set.len(), 1);

    // Difference with empty
    let diff = set.difference(&empty);
    assert_eq!(diff.len(), 1);

    // Empty is subset of everything
    assert!(empty.is_subset_of(&set));
}

#[test]
fn test_support_set_from_iterator() {
    let ids = vec![gates::H, gates::X, gates::CX];
    let set: GateSupportSet = ids.into_iter().collect();

    assert_eq!(set.len(), 3);
    assert!(set.contains(gates::H));
    assert!(set.contains(gates::X));
    assert!(set.contains(gates::CX));
}

#[test]
fn test_support_set_intersect() {
    let mut set1 = GateSupportSet::new();
    set1.insert(gates::H);
    set1.insert(gates::X);
    set1.insert(gates::CX);

    let mut set2 = GateSupportSet::new();
    set2.insert(gates::X);
    set2.insert(gates::CX);
    set2.insert(gates::CZ);

    set1.intersect_with(&set2);

    assert!(!set1.contains(gates::H)); // Only in set1
    assert!(set1.contains(gates::X)); // In both
    assert!(set1.contains(gates::CX)); // In both
    assert!(!set1.contains(gates::CZ)); // Only in set2
    assert_eq!(set1.len(), 2);
}

#[test]
fn test_canonicalizer_custom_rule() {
    let mut canon = GateCanonicalizer::new();

    // Add a custom canonicalization for a user-defined gate
    let custom_rot = GateId(256);
    let custom_fixed = GateId(257);

    canon.add(custom_rot, Angle64::QUARTER_TURN, custom_fixed);

    assert_eq!(
        canon.canonicalize(custom_rot, &[Angle64::QUARTER_TURN]),
        Some(custom_fixed)
    );

    // Other angles should not canonicalize
    assert_eq!(canon.canonicalize(custom_rot, &[Angle64::HALF_TURN]), None);
}

#[test]
fn test_canonicalizer_expand() {
    let canon = GateCanonicalizer::standard();

    // SZ should expand to RZ(π/2)
    let (gate, angle) = canon.expand(gates::SZ).unwrap();
    assert_eq!(gate, gates::RZ);
    assert_eq!(angle, Angle64::QUARTER_TURN);

    // T should expand to RZ(π/4)
    let (gate, angle) = canon.expand(gates::T).unwrap();
    assert_eq!(gate, gates::RZ);
    assert_eq!(angle, Angle64::HALF_TURN / 4);
}

#[test]
fn test_canonicalizer_expand_not_found() {
    let canon = GateCanonicalizer::standard();

    // CX has no expansion (not a parameterized gate)
    assert!(canon.expand(gates::CX).is_none());

    // Random user gate has no expansion
    assert!(canon.expand(GateId(500)).is_none());
}

#[test]
fn test_canonicalizer_can_canonicalize() {
    let canon = GateCanonicalizer::standard();

    assert!(canon.can_canonicalize(gates::RZ));
    assert!(canon.can_canonicalize(gates::RX));
    assert!(canon.can_canonicalize(gates::RY));
    assert!(!canon.can_canonicalize(gates::H));
    assert!(!canon.can_canonicalize(gates::CX));
}

#[test]
fn test_canonicalizer_get_forms_for() {
    let canon = GateCanonicalizer::standard();

    let rz_forms = canon.get_forms_for(gates::RZ);

    // RZ has several canonical forms
    assert!(rz_forms.len() >= 4); // 0, π/4, π/2, π at minimum

    // Check one of them
    let sz_form = rz_forms.iter().find(|f| f.to_gate == gates::SZ);
    assert!(sz_form.is_some());
    assert_eq!(sz_form.unwrap().angle, Angle64::QUARTER_TURN);
}

#[test]
fn test_gate_spec_builder_pattern() {
    let spec = GateSpec::new("TestGate")
        .with_quantum_arity(2)
        .with_angle_arity(1)
        .with_param_arity(0)
        .with_returns_result(false)
        .with_category(GateCategory::TwoQubitUnitary);

    assert_eq!(spec.name, "TestGate");
    assert_eq!(spec.quantum_arity, 2);
    assert_eq!(spec.angle_arity, 1);
    assert!(!spec.returns_result);
    assert!(spec.is_two_qubit());
    assert!(spec.is_parameterized());
}

#[test]
fn test_gate_spec_is_methods() {
    let sq = GateSpec::new("SQ").with_quantum_arity(1);
    assert!(sq.is_single_qubit());
    assert!(!sq.is_two_qubit());
    assert!(!sq.is_parameterized());

    let tq = GateSpec::new("TQ").with_quantum_arity(2);
    assert!(!tq.is_single_qubit());
    assert!(tq.is_two_qubit());

    let param = GateSpec::new("Param").with_angle_arity(2);
    assert!(param.is_parameterized());
}

#[test]
fn test_gate_category_custom() {
    assert_ne!(GateCategory::Custom(0), GateCategory::Custom(1));
    assert_ne!(GateCategory::Custom(0), GateCategory::SingleQubitUnitary);

    // Custom categories should be distinguishable
    let cat1 = GateCategory::Custom(42);
    let cat2 = GateCategory::Custom(42);
    assert_eq!(cat1, cat2);
}

#[test]
fn test_registry_lookup_case_sensitive() {
    let registry = GateRegistry::new();

    // Lookup is case-sensitive
    assert!(registry.lookup("H").is_some());
    assert!(registry.lookup("h").is_none());
    assert!(registry.lookup("CX").is_some());
    assert!(registry.lookup("cx").is_none());
    assert!(registry.lookup("Cx").is_none());
}

#[test]
fn test_all_core_gates_have_correct_arity() {
    let registry = GateRegistry::new();

    // Single-qubit gates
    for id in [
        gates::I,
        gates::X,
        gates::Y,
        gates::Z,
        gates::H,
        gates::SX,
        gates::SY,
        gates::SZ,
        gates::T,
        gates::RX,
        gates::RY,
        gates::RZ,
    ] {
        let spec = registry.get(id).unwrap();
        assert_eq!(
            spec.quantum_arity, 1,
            "Gate {} should have arity 1",
            spec.name
        );
    }

    // Two-qubit gates
    for id in [
        gates::CX,
        gates::CY,
        gates::CZ,
        gates::SWAP,
        gates::SZZ,
        gates::RZZ,
    ] {
        let spec = registry.get(id).unwrap();
        assert_eq!(
            spec.quantum_arity, 2,
            "Gate {} should have arity 2",
            spec.name
        );
    }

    // Three-qubit gates
    for id in [gates::CCX, gates::CCZ, gates::CSWAP] {
        let spec = registry.get(id).unwrap();
        assert_eq!(
            spec.quantum_arity, 3,
            "Gate {} should have arity 3",
            spec.name
        );
    }
}

#[test]
fn test_parameterized_gates_have_angle_arity() {
    let registry = GateRegistry::new();

    // Single-angle gates
    for id in [
        gates::RX,
        gates::RY,
        gates::RZ,
        gates::RZZ,
        gates::RXX,
        gates::RYY,
    ] {
        let spec = registry.get(id).unwrap();
        assert_eq!(
            spec.angle_arity, 1,
            "Gate {} should have angle_arity 1",
            spec.name
        );
    }

    // Non-parameterized gates
    for id in [gates::H, gates::X, gates::CX, gates::CZ, gates::T] {
        let spec = registry.get(id).unwrap();
        assert_eq!(
            spec.angle_arity, 0,
            "Gate {} should have angle_arity 0",
            spec.name
        );
    }
}

#[test]
fn test_measurement_gates_return_result() {
    let registry = GateRegistry::new();

    for id in [gates::MZ, gates::MEASURE_LEAKED, gates::MEASURE_FREE] {
        let spec = registry.get(id).unwrap();
        assert!(
            spec.returns_result,
            "Gate {} should return result",
            spec.name
        );
    }

    // Non-measurement gates should not return result
    for id in [gates::H, gates::CX, gates::PZ, gates::RZ] {
        let spec = registry.get(id).unwrap();
        assert!(
            !spec.returns_result,
            "Gate {} should not return result",
            spec.name
        );
    }
}

// ============================================================================
// AngleSnapper Tests
// ============================================================================

#[test]
fn test_snapper_exact_angle_no_change() {
    let snapper = AngleSnapper::standard(1e-9);

    let result = snapper.snap(Angle64::QUARTER_TURN).unwrap();
    assert_eq!(result.snapped, Angle64::QUARTER_TURN);
    assert_eq!(result.distance, 0.0);
}

#[test]
fn test_snapper_close_angle() {
    let snapper = AngleSnapper::standard(1e-6);

    // Slightly off from pi/2 (0.25 turns)
    let close = Angle64::from_turns(0.25 + 1e-9);
    let result = snapper.snap(close).unwrap();

    assert_eq!(result.snapped, Angle64::QUARTER_TURN);
    assert!(result.distance < 1e-6);
}

#[test]
fn test_snapper_fails_outside_tolerance() {
    let snapper = AngleSnapper::standard(1e-9);

    // Way off from any standard angle
    let far = Angle64::from_turns(0.123_456);
    let result = snapper.snap(far);

    assert!(result.is_err());
}

#[test]
fn test_snapper_snap_or_keep() {
    let snapper = AngleSnapper::standard(1e-9);

    // Exact angle gets snapped (to itself)
    assert_eq!(
        snapper.snap_or_keep(Angle64::QUARTER_TURN),
        Angle64::QUARTER_TURN
    );

    // Far angle gets kept
    let arbitrary = Angle64::from_turns(0.123);
    assert_eq!(snapper.snap_or_keep(arbitrary), arbitrary);
}

#[test]
fn test_snapper_clifford() {
    let snapper = AngleSnapper::clifford(1e-9);

    // Clifford angles should snap
    assert!(snapper.snap(Angle64::ZERO).is_ok());
    assert!(snapper.snap(Angle64::QUARTER_TURN).is_ok());
    assert!(snapper.snap(Angle64::HALF_TURN).is_ok());
    assert!(snapper.snap(Angle64::THREE_QUARTERS_TURN).is_ok());

    // T angle (pi/4) should NOT snap with Clifford snapper
    let t_angle = Angle64::HALF_TURN / 4;
    assert!(snapper.snap(t_angle).is_err());
}

#[test]
fn test_snapper_add_target() {
    let mut snapper = AngleSnapper::clifford(1e-9);

    // T angle doesn't snap initially
    let t_angle = Angle64::HALF_TURN / 4;
    assert!(snapper.snap(t_angle).is_err());

    // Add T angle as target
    snapper.add_target(t_angle);

    // Now it should snap
    assert!(snapper.snap(t_angle).is_ok());
}

#[test]
fn test_snap_policy_exact() {
    let snapper = AngleSnapper::standard(1e-9);
    let policy = SnapPolicy::Exact;

    // Exact policy returns angle unchanged
    let arbitrary = Angle64::from_turns(0.123);
    let result = policy.apply(arbitrary, &snapper).unwrap();
    assert_eq!(result, arbitrary);
}

#[test]
fn test_snap_policy_snap_or_fail() {
    let snapper = AngleSnapper::standard(1e-9);
    let policy = SnapPolicy::snap_or_fail(1e-9);

    // Close angle succeeds
    let close = Angle64::from_turns(0.25);
    assert!(policy.apply(close, &snapper).is_ok());

    // Far angle fails
    let far = Angle64::from_turns(0.123);
    assert!(policy.apply(far, &snapper).is_err());
}

#[test]
fn test_snap_policy_snap_or_keep() {
    let snapper = AngleSnapper::standard(1e-9);
    let policy = SnapPolicy::snap_or_keep(1e-9);

    // Close angle gets snapped
    let close = Angle64::from_turns(0.25);
    let result = policy.apply(close, &snapper).unwrap();
    assert_eq!(result, Angle64::QUARTER_TURN);

    // Far angle is kept
    let far = Angle64::from_turns(0.123);
    let result = policy.apply(far, &snapper).unwrap();
    assert_eq!(result, far);
}

// ============================================================================
// CircuitValidator Tests
// ============================================================================

fn make_gate(id: GateId, angles: &[Angle64]) -> GateForValidation {
    GateForValidation {
        gate_id: id,
        angles: angles.to_vec(),
    }
}

#[test]
fn test_clifford_validator_accepts_clifford_circuit() {
    let validator = CliffordValidator::new();
    let registry = GateRegistry::new();

    let circuit = vec![
        make_gate(gates::H, &[]),
        make_gate(gates::CX, &[]),
        make_gate(gates::SZ, &[]),
        make_gate(gates::MZ, &[]),
    ];

    assert!(validator.validate(&circuit, &registry).is_ok());
}

#[test]
fn test_clifford_validator_rejects_t_gate() {
    let validator = CliffordValidator::new();
    let registry = GateRegistry::new();

    let circuit = vec![
        make_gate(gates::H, &[]),
        make_gate(gates::T, &[]), // Not Clifford!
    ];

    let result = validator.validate(&circuit, &registry);
    assert!(matches!(result, Err(ValidationError::ForbiddenGate { .. })));
}

#[test]
fn test_clifford_validator_rejects_arbitrary_rz() {
    let validator = CliffordValidator::new();
    let registry = GateRegistry::new();

    let circuit = vec![
        make_gate(gates::RZ, &[Angle64::from_turns(0.123)]), // Arbitrary angle
    ];

    let result = validator.validate(&circuit, &registry);
    assert!(matches!(
        result,
        Err(ValidationError::ForbiddenAngle { .. })
    ));
}

#[test]
fn test_clifford_validator_accepts_rz_at_clifford_angle() {
    let validator = CliffordValidator::new();
    let registry = GateRegistry::new();

    let circuit = vec![
        make_gate(gates::RZ, &[Angle64::QUARTER_TURN]), // pi/2 is Clifford
    ];

    assert!(validator.validate(&circuit, &registry).is_ok());
}

#[test]
fn test_clifford_t_validator_accepts_t() {
    let validator = CliffordTValidator::new();
    let registry = GateRegistry::new();

    let circuit = vec![
        make_gate(gates::H, &[]),
        make_gate(gates::T, &[]),
        make_gate(gates::CX, &[]),
    ];

    assert!(validator.validate(&circuit, &registry).is_ok());
}

#[test]
fn test_clifford_t_validator_accepts_rz_at_t_angle() {
    let validator = CliffordTValidator::new();
    let registry = GateRegistry::new();

    let t_angle = Angle64::HALF_TURN / 4; // pi/4

    let circuit = vec![make_gate(gates::RZ, &[t_angle])];

    assert!(validator.validate(&circuit, &registry).is_ok());
}

#[test]
fn test_exact_angle_validator_accepts_canonicalizable() {
    let validator = ExactAngleValidator::new();
    let registry = GateRegistry::new();

    let circuit = vec![
        make_gate(gates::RZ, &[Angle64::QUARTER_TURN]), // -> SZ
        make_gate(gates::RZ, &[Angle64::HALF_TURN / 4]), // -> T
    ];

    assert!(validator.validate(&circuit, &registry).is_ok());
}

#[test]
fn test_exact_angle_validator_rejects_non_canonicalizable() {
    let validator = ExactAngleValidator::new();
    let registry = GateRegistry::new();

    let circuit = vec![
        make_gate(gates::RZ, &[Angle64::from_turns(0.123)]), // Can't canonicalize
    ];

    let result = validator.validate(&circuit, &registry);
    assert!(matches!(
        result,
        Err(ValidationError::NonCanonicalAngle { .. })
    ));
}

#[test]
fn test_exact_angle_validator_accepts_non_parameterized() {
    let validator = ExactAngleValidator::new();
    let registry = GateRegistry::new();

    let circuit = vec![make_gate(gates::H, &[]), make_gate(gates::CX, &[])];

    assert!(validator.validate(&circuit, &registry).is_ok());
}

#[test]
fn test_allow_list_validator() {
    let mut validator = AllowListValidator::new();
    validator.allow(gates::H);
    validator.allow(gates::CX);

    let registry = GateRegistry::new();

    // Allowed gates pass
    let circuit = vec![make_gate(gates::H, &[]), make_gate(gates::CX, &[])];
    assert!(validator.validate(&circuit, &registry).is_ok());

    // Disallowed gate fails
    let circuit = vec![
        make_gate(gates::H, &[]),
        make_gate(gates::RZ, &[Angle64::QUARTER_TURN]), // Not in allow list
    ];
    let result = validator.validate(&circuit, &registry);
    assert!(matches!(result, Err(ValidationError::ForbiddenGate { .. })));
}

#[test]
fn test_composite_validator() {
    let validator = CompositeValidator::new()
        .with(CliffordValidator::new())
        .with(ExactAngleValidator::new());

    let registry = GateRegistry::new();

    // Must pass both: Clifford AND exact angles
    let circuit = vec![
        make_gate(gates::RZ, &[Angle64::QUARTER_TURN]), // Clifford angle, canonicalizable
    ];
    assert!(validator.validate(&circuit, &registry).is_ok());

    // Fails Clifford check (T is not Clifford)
    let circuit = vec![make_gate(gates::T, &[])];
    let result = validator.validate(&circuit, &registry);
    assert!(matches!(result, Err(ValidationError::ForbiddenGate { .. })));
}

#[test]
fn test_validator_is_gate_allowed() {
    let validator = CliffordValidator::new();
    let registry = GateRegistry::new();

    assert!(validator.is_gate_allowed(gates::H, &[], &registry));
    assert!(validator.is_gate_allowed(gates::CX, &[], &registry));
    assert!(validator.is_gate_allowed(gates::RZ, &[Angle64::QUARTER_TURN], &registry));
    assert!(!validator.is_gate_allowed(gates::T, &[], &registry));
    assert!(!validator.is_gate_allowed(gates::RZ, &[Angle64::from_turns(0.1)], &registry));
}

#[test]
fn test_validation_error_display() {
    let err = ValidationError::ForbiddenGate {
        gate_id: gates::T,
        gate_name: "T".to_string(),
        position: 5,
    };
    let msg = format!("{err}");
    assert!(msg.contains('T'));
    assert!(msg.contains('5'));

    let err = ValidationError::NonCanonicalAngle {
        gate_id: gates::RZ,
        gate_name: "RZ".to_string(),
        angle: Angle64::from_turns(0.123),
        position: 3,
    };
    let msg = format!("{err}");
    assert!(msg.contains("RZ"));
    assert!(msg.contains('3'));
}

// ============================================================================
// GateAdaptor Tests
// ============================================================================

#[test]
fn test_standard_adaptor_can_adapt() {
    let adaptor = StandardAdaptor::stab_vec();

    assert!(adaptor.can_adapt(gates::T));
    assert!(adaptor.can_adapt(gates::Tdg));
    assert!(adaptor.can_adapt(gates::RX));
    assert!(adaptor.can_adapt(gates::RY));
    assert!(adaptor.can_adapt(gates::SWAP));
    assert!(adaptor.can_adapt(gates::RZZ));
    assert!(adaptor.can_adapt(gates::CCX));

    // These should NOT be adaptable (they're already in target set)
    assert!(!adaptor.can_adapt(gates::H));
    assert!(!adaptor.can_adapt(gates::CX));
    assert!(!adaptor.can_adapt(gates::RZ));
}

#[test]
fn test_standard_adaptor_t_gate() {
    let adaptor = StandardAdaptor::stab_vec();

    let result = adaptor.adapt(gates::T, &[QubitId(0)], &[]);

    // T = RZ(π/4)
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].gate_id, gates::RZ);
    assert_eq!(result[0].qubits[0], QubitId(0));
    assert_eq!(result[0].angles[0], Angle64::HALF_TURN / 4);
}

#[test]
fn test_standard_adaptor_tdg_gate() {
    let adaptor = StandardAdaptor::stab_vec();

    let result = adaptor.adapt(gates::Tdg, &[QubitId(0)], &[]);

    // Tdg = RZ(-π/4)
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].gate_id, gates::RZ);
    let expected_angle = Angle64::ZERO - Angle64::HALF_TURN / 4;
    assert_eq!(result[0].angles[0], expected_angle);
}

#[test]
fn test_standard_adaptor_rx_gate() {
    let adaptor = StandardAdaptor::stab_vec();

    let theta = Angle64::QUARTER_TURN;
    let result = adaptor.adapt(gates::RX, &[QubitId(0)], &[theta]);

    // RX(θ) = H RZ(θ) H
    assert_eq!(result.len(), 3);
    assert_eq!(result[0].gate_id, gates::H);
    assert_eq!(result[1].gate_id, gates::RZ);
    assert_eq!(result[1].angles[0], theta);
    assert_eq!(result[2].gate_id, gates::H);
}

#[test]
fn test_standard_adaptor_swap_gate() {
    let adaptor = StandardAdaptor::stab_vec();

    let result = adaptor.adapt(gates::SWAP, &[QubitId(0), QubitId(1)], &[]);

    // SWAP = CX CX CX
    assert_eq!(result.len(), 3);
    assert!(result.iter().all(|g| g.gate_id == gates::CX));
}

#[test]
fn test_standard_adaptor_rzz_gate() {
    let adaptor = StandardAdaptor::stab_vec();

    let theta = Angle64::QUARTER_TURN;
    let result = adaptor.adapt(gates::RZZ, &[QubitId(0), QubitId(1)], &[theta]);

    // RZZ(θ) = CX RZ(θ) CX
    assert_eq!(result.len(), 3);
    assert_eq!(result[0].gate_id, gates::CX);
    assert_eq!(result[1].gate_id, gates::RZ);
    assert_eq!(result[1].angles[0], theta);
    assert_eq!(result[2].gate_id, gates::CX);
}

#[test]
fn test_standard_adaptor_ccx_gate() {
    let adaptor = StandardAdaptor::stab_vec();

    let result = adaptor.adapt(gates::CCX, &[QubitId(0), QubitId(1), QubitId(2)], &[]);

    // CCX decomposes to ~15 gates
    assert!(result.len() > 10);

    // Should contain H, CX, and RZ gates
    assert!(result.iter().any(|g| g.gate_id == gates::H));
    assert!(result.iter().any(|g| g.gate_id == gates::CX));
    assert!(result.iter().any(|g| g.gate_id == gates::RZ));
}

#[test]
fn test_composite_adaptor() {
    let adaptor = CompositeAdaptor::new().with(StandardAdaptor::stab_vec());

    assert!(adaptor.can_adapt(gates::T));
    assert!(adaptor.can_adapt(gates::SWAP));

    let result = adaptor.adapt(gates::T, &[QubitId(0)], &[]);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].gate_id, gates::RZ);
}

#[test]
fn test_custom_adaptor() {
    let custom_gate = GateId(256);

    let adaptor = CustomAdaptor::new(custom_gate, |qubits, angles| {
        // Custom decomposition: just wrap in H gates
        vec![
            AdaptedGate::single(gates::H, qubits[0]),
            AdaptedGate::rotation(gates::RZ, qubits[0], angles[0]),
            AdaptedGate::single(gates::H, qubits[0]),
        ]
    });

    assert!(adaptor.can_adapt(custom_gate));
    assert!(!adaptor.can_adapt(gates::H));

    let theta = Angle64::QUARTER_TURN;
    let result = adaptor.adapt(custom_gate, &[QubitId(0)], &[theta]);

    assert_eq!(result.len(), 3);
    assert_eq!(result[0].gate_id, gates::H);
    assert_eq!(result[1].gate_id, gates::RZ);
    assert_eq!(result[1].angles[0], theta);
    assert_eq!(result[2].gate_id, gates::H);
}

#[test]
fn test_adaptor_adaptable_gates() {
    let adaptor = StandardAdaptor::stab_vec();
    let adaptable = adaptor.adaptable_gates();

    assert!(adaptable.contains(gates::T));
    assert!(adaptable.contains(gates::SWAP));
    assert!(!adaptable.contains(gates::H));
}

#[test]
fn test_adapted_gate_constructors() {
    let single = AdaptedGate::single(gates::H, QubitId(0));
    assert_eq!(single.gate_id, gates::H);
    assert_eq!(single.qubits.len(), 1);
    assert!(single.angles.is_empty());

    let rotation = AdaptedGate::rotation(gates::RZ, QubitId(1), Angle64::QUARTER_TURN);
    assert_eq!(rotation.gate_id, gates::RZ);
    assert_eq!(rotation.qubits[0], QubitId(1));
    assert_eq!(rotation.angles[0], Angle64::QUARTER_TURN);

    let two_qubit = AdaptedGate::two_qubit(gates::CX, QubitId(0), QubitId(1));
    assert_eq!(two_qubit.gate_id, gates::CX);
    assert_eq!(two_qubit.qubits.len(), 2);
}
