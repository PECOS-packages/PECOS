# Extensible Gates Test Plan

This document outlines the tests needed to validate the extensible gates implementation.

## 1. GateId and GateSpec Tests

### 1.1 GateId Basics
```rust
#[test]
fn test_gate_id_core_range() {
    // IDs 0-255 are core gates
    assert!(GateId(0).is_core());
    assert!(GateId(255).is_core());
    assert!(!GateId(256).is_core());
    assert!(GateId(256).is_user_defined());
}

#[test]
fn test_gate_id_size() {
    // Must be u16 for compact storage
    assert_eq!(std::mem::size_of::<GateId>(), 2);
}

#[test]
fn test_gate_id_constants_match_enum() {
    // Verify gate ID constants match current GateType enum values
    assert_eq!(gates::X, GateId(GateType::X as u16));
    assert_eq!(gates::H, GateId(GateType::H as u16));
    assert_eq!(gates::CX, GateId(GateType::CX as u16));
    // ... all core gates
}
```

### 1.2 GateSpec
```rust
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

    assert_eq!(spec.quantum_arity, 2);
    assert_eq!(spec.angle_arity, 3);
}

#[test]
fn test_gate_spec_category_classification() {
    // Verify category correctly classifies gates
    let sq = GateSpec { category: GateCategory::SingleQubitUnitary, ..default() };
    let tq = GateSpec { category: GateCategory::TwoQubitUnitary, ..default() };
    let meas = GateSpec { category: GateCategory::Measurement, ..default() };

    assert!(sq.category.is_unitary());
    assert!(tq.category.is_unitary());
    assert!(!meas.category.is_unitary());
    assert!(meas.category.returns_result());
}
```

## 2. GateRegistry Tests

### 2.1 Core Gate Initialization
```rust
#[test]
fn test_registry_has_core_gates() {
    let registry = GateRegistry::new();

    // All core gates should be pre-registered
    assert!(registry.get(gates::X).is_some());
    assert!(registry.get(gates::H).is_some());
    assert!(registry.get(gates::CX).is_some());
    assert!(registry.get(gates::MZ).is_some());
}

#[test]
fn test_registry_core_gate_specs_correct() {
    let registry = GateRegistry::new();

    let h = registry.get(gates::H).unwrap();
    assert_eq!(h.name, "H");
    assert_eq!(h.quantum_arity, 1);
    assert_eq!(h.angle_arity, 0);

    let cx = registry.get(gates::CX).unwrap();
    assert_eq!(cx.quantum_arity, 2);

    let rz = registry.get(gates::RZ).unwrap();
    assert_eq!(rz.angle_arity, 1);
}
```

### 2.2 User Gate Registration
```rust
#[test]
fn test_register_user_gate() {
    let mut registry = GateRegistry::new();

    let id = registry.register(GateSpec {
        name: "MyRotation",
        quantum_arity: 2,
        angle_arity: 3,
        ..default()
    });

    assert!(id.is_user_defined());
    assert_eq!(id.0, 256); // First user gate

    let spec = registry.get(id).unwrap();
    assert_eq!(spec.name, "MyRotation");
}

#[test]
fn test_register_multiple_user_gates() {
    let mut registry = GateRegistry::new();

    let id1 = registry.register(GateSpec { name: "Gate1", ..default() });
    let id2 = registry.register(GateSpec { name: "Gate2", ..default() });
    let id3 = registry.register(GateSpec { name: "Gate3", ..default() });

    assert_eq!(id1.0, 256);
    assert_eq!(id2.0, 257);
    assert_eq!(id3.0, 258);
}

#[test]
fn test_lookup_by_name() {
    let mut registry = GateRegistry::new();
    registry.register(GateSpec { name: "MyGate", ..default() });

    assert_eq!(registry.lookup("H"), Some(gates::H));
    assert_eq!(registry.lookup("MyGate"), Some(GateId(256)));
    assert_eq!(registry.lookup("NonExistent"), None);
}
```

### 2.3 Registry Scoping (Not Global)
```rust
#[test]
fn test_registries_are_independent() {
    let mut registry1 = GateRegistry::new();
    let mut registry2 = GateRegistry::new();

    let id1 = registry1.register(GateSpec { name: "GateA", ..default() });
    let id2 = registry2.register(GateSpec { name: "GateB", ..default() });

    // Same ID value but different gates
    assert_eq!(id1.0, id2.0); // Both 256

    // But specs are different in each registry
    assert_eq!(registry1.get(id1).unwrap().name, "GateA");
    assert_eq!(registry2.get(id2).unwrap().name, "GateB");

    // Registry1 doesn't know about GateB
    assert!(registry1.lookup("GateB").is_none());
}
```

## 3. Const Table Tests (Core Gate Metadata)

```rust
#[test]
fn test_core_quantum_arity_table() {
    // Verify compile-time const table matches runtime values
    let registry = GateRegistry::new();

    for id in 0..256u16 {
        let gate_id = GateId(id);
        if let Some(spec) = registry.get(gate_id) {
            assert_eq!(
                CORE_QUANTUM_ARITY[id as usize],
                spec.quantum_arity,
                "Mismatch for gate {:?}", spec.name
            );
        }
    }
}

#[test]
fn test_core_angle_arity_table() {
    let registry = GateRegistry::new();

    for id in 0..256u16 {
        let gate_id = GateId(id);
        if let Some(spec) = registry.get(gate_id) {
            assert_eq!(
                CORE_ANGLE_ARITY[id as usize],
                spec.angle_arity,
                "Mismatch for gate {:?}", spec.name
            );
        }
    }
}

#[test]
fn test_core_category_table() {
    let registry = GateRegistry::new();

    for id in 0..256u16 {
        let gate_id = GateId(id);
        if let Some(spec) = registry.get(gate_id) {
            assert_eq!(
                CORE_CATEGORY[id as usize],
                spec.category,
                "Mismatch for gate {:?}", spec.name
            );
        }
    }
}
```

## 4. BitVec Support Set Tests

### 4.1 Basic Operations
```rust
#[test]
fn test_gate_support_set_insert_contains() {
    let mut set = GateSupportSet::new();

    set.add(gates::H);
    set.add(gates::CX);

    assert!(set.supports(gates::H));
    assert!(set.supports(gates::CX));
    assert!(!set.supports(gates::RZ));
}

#[test]
fn test_gate_support_set_user_gates() {
    let mut set = GateSupportSet::new();

    let user_gate = GateId(300);
    set.add(user_gate);

    assert!(set.supports(user_gate));
    assert!(!set.supports(GateId(301)));
}

#[test]
fn test_gate_support_set_union() {
    let mut set1 = GateSupportSet::new();
    set1.add(gates::H);
    set1.add(gates::X);

    let mut set2 = GateSupportSet::new();
    set2.add(gates::CX);
    set2.add(gates::X); // Overlap

    set1.union(&set2);

    assert!(set1.supports(gates::H));
    assert!(set1.supports(gates::X));
    assert!(set1.supports(gates::CX));
}
```

### 4.2 Set Operations for Validation
```rust
#[test]
fn test_unsupported_gates_detection() {
    let mut required = GateSupportSet::new();
    required.add(gates::H);
    required.add(gates::CX);
    required.add(gates::RZ);

    let mut supported = GateSupportSet::new();
    supported.add(gates::H);
    supported.add(gates::CX);
    // RZ not supported

    let unsupported = required.difference(&supported);

    assert!(!unsupported.supports(gates::H));
    assert!(!unsupported.supports(gates::CX));
    assert!(unsupported.supports(gates::RZ)); // This is unsupported
}
```

## 5. Gate Canonicalization Tests

### 5.1 Exact Angle Matching
```rust
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
fn test_canonicalize_t_gate() {
    let canon = GateCanonicalizer::standard();

    // RZ(π/4) = T
    let result = canon.canonicalize(gates::RZ, &[Angle64::HALF_TURN / 4]);
    assert_eq!(result, Some(gates::T));
}

#[test]
fn test_canonicalize_negative_angles() {
    let canon = GateCanonicalizer::standard();

    // RZ(-π/2) = SZdg
    let result = canon.canonicalize(gates::RZ, &[Angle64::ZERO - Angle64::QUARTER_TURN]);
    assert_eq!(result, Some(gates::SZdg));
}

#[test]
fn test_no_canonicalization_for_arbitrary_angle() {
    let canon = GateCanonicalizer::standard();

    // RZ(0.123) has no canonical form
    let arbitrary = Angle64::from_turns(0.123);
    let result = canon.canonicalize(gates::RZ, &[arbitrary]);
    assert_eq!(result, None);
}
```

### 5.2 Angle64 Exactness
```rust
#[test]
fn test_angle64_exact_comparison() {
    // Standard angles are exactly representable
    let quarter = Angle64::QUARTER_TURN;
    let from_turns = Angle64::from_turns(0.25);

    assert_eq!(quarter, from_turns); // EXACT equality
}

#[test]
fn test_angle64_from_turns_exact_fractions() {
    // Powers of 2 denominators are exact
    assert_eq!(Angle64::from_turns(0.5), Angle64::HALF_TURN);
    assert_eq!(Angle64::from_turns(0.25), Angle64::QUARTER_TURN);
    assert_eq!(Angle64::from_turns(0.125), Angle64::HALF_TURN / 4);
    assert_eq!(Angle64::from_turns(0.0625), Angle64::HALF_TURN / 8);
}

#[test]
fn test_angle64_arithmetic_exactness() {
    // Arithmetic on exact angles stays exact
    let t = Angle64::HALF_TURN / 4; // π/4
    let s = t + t; // π/2

    assert_eq!(s, Angle64::QUARTER_TURN);
}
```

## 6. Angle Snapping Tests

### 6.1 Basic Snapping
```rust
#[test]
fn test_snap_exact_angle_no_change() {
    let snapper = AngleSnapper::standard(1e-9);

    let result = snapper.snap(Angle64::QUARTER_TURN).unwrap();
    assert_eq!(result.snapped, Angle64::QUARTER_TURN);
    assert_eq!(result.distance, 0.0);
}

#[test]
fn test_snap_close_angle() {
    let snapper = AngleSnapper::standard(1e-9);

    // Slightly off from π/2
    let close = Angle64::from_turns(0.25 + 1e-12);
    let result = snapper.snap(close).unwrap();

    assert_eq!(result.snapped, Angle64::QUARTER_TURN);
    assert!(result.distance < 1e-9);
}

#[test]
fn test_snap_fails_outside_tolerance() {
    let snapper = AngleSnapper::standard(1e-9);

    // Way off from any standard angle
    let far = Angle64::from_turns(0.123456);
    let result = snapper.snap(far);

    assert!(result.is_err());
}
```

### 6.2 Snapping Policies
```rust
#[test]
fn test_snap_or_keep_keeps_non_standard() {
    let snapper = AngleSnapper::standard(1e-9);

    let arbitrary = Angle64::from_turns(0.123);
    let result = snapper.snap_or_keep(arbitrary);

    assert_eq!(result, arbitrary); // Kept as-is
}

#[test]
fn test_clifford_snapper_rejects_t() {
    let snapper = AngleSnapper::clifford(1e-9);

    // T gate angle (π/4) is not a Clifford angle
    let t_angle = Angle64::HALF_TURN / 4;
    let result = snapper.snap(t_angle);

    assert!(result.is_err());
}
```

## 7. Circuit Validator Tests

### 7.1 Clifford Validator
```rust
#[test]
fn test_clifford_validator_accepts_clifford_circuit() {
    let validator = CliffordValidator::new();
    let registry = GateRegistry::new();

    let circuit = Circuit::from_gates(vec![
        Gate::h(&[0]),
        Gate::cx(&[0, 1]),
        Gate::sz(&[0]),
        Gate::measure(&[0, 1]),
    ]);

    assert!(validator.validate(&circuit, &registry).is_ok());
}

#[test]
fn test_clifford_validator_rejects_t_gate() {
    let validator = CliffordValidator::new();
    let registry = GateRegistry::new();

    let circuit = Circuit::from_gates(vec![
        Gate::h(&[0]),
        Gate::t(&[0]), // Not Clifford!
    ]);

    let result = validator.validate(&circuit, &registry);
    assert!(matches!(result, Err(ValidationError::ForbiddenGate { .. })));
}

#[test]
fn test_clifford_validator_rejects_arbitrary_rz() {
    let validator = CliffordValidator::new();
    let registry = GateRegistry::new();

    let circuit = Circuit::from_gates(vec![
        Gate::rz(Angle64::from_turns(0.123), &[0]), // Arbitrary angle
    ]);

    let result = validator.validate(&circuit, &registry);
    assert!(matches!(result, Err(ValidationError::ForbiddenAngle { .. })));
}

#[test]
fn test_clifford_validator_accepts_rz_at_clifford_angle() {
    let validator = CliffordValidator::new();
    let registry = GateRegistry::new();

    let circuit = Circuit::from_gates(vec![
        Gate::rz(Angle64::QUARTER_TURN, &[0]), // π/2 is Clifford
    ]);

    assert!(validator.validate(&circuit, &registry).is_ok());
}
```

### 7.2 Exact Angle Validator
```rust
#[test]
fn test_exact_angle_validator_accepts_canonicalizable() {
    let validator = ExactAngleValidator::new();
    let registry = GateRegistry::new();

    let circuit = Circuit::from_gates(vec![
        Gate::rz(Angle64::QUARTER_TURN, &[0]),    // → SZ
        Gate::rz(Angle64::HALF_TURN / 4, &[0]),   // → T
    ]);

    assert!(validator.validate(&circuit, &registry).is_ok());
}

#[test]
fn test_exact_angle_validator_rejects_non_canonicalizable() {
    let validator = ExactAngleValidator::new();
    let registry = GateRegistry::new();

    let circuit = Circuit::from_gates(vec![
        Gate::rz(Angle64::from_turns(0.123), &[0]), // Can't canonicalize
    ]);

    let result = validator.validate(&circuit, &registry);
    assert!(matches!(result, Err(ValidationError::NonCanonicalAngle { .. })));
}
```

## 8. Build-Time Validation Tests

### 8.1 Simulator Support Validation
```rust
#[test]
fn test_build_fails_for_unsupported_gate() {
    let circuit = Circuit::from_gates(vec![
        Gate::h(&[0]),
        Gate::custom(GateId(256), &[0, 1], &[]), // Unregistered custom gate
    ]);

    let result = ToolBuilder::new()
        .with_program(circuit)
        .with_simulator(SparseStab::new(2))
        .build();

    assert!(matches!(result, Err(ValidationError::UnknownGateId(GateId(256)))));
}

#[test]
fn test_build_fails_for_unsupported_by_simulator() {
    let mut registry = GateRegistry::new();
    let custom = registry.register(GateSpec {
        name: "CustomGate",
        quantum_arity: 2,
        ..default()
    });

    let circuit = Circuit::from_gates(vec![
        Gate::h(&[0]),
        Gate::custom(custom, &[0, 1], &[]),
    ]);

    // No adaptor for CustomGate
    let result = ToolBuilder::new()
        .with_program(circuit)
        .with_simulator(SparseStab::new(2))
        .with_gate_registry(registry)
        .build();

    assert!(matches!(result, Err(ValidationError::UnsupportedGates(_))));
}

#[test]
fn test_build_succeeds_with_adaptor() {
    let mut registry = GateRegistry::new();
    let custom = registry.register(GateSpec {
        name: "CustomGate",
        quantum_arity: 2,
        ..default()
    });

    let circuit = Circuit::from_gates(vec![
        Gate::custom(custom, &[0, 1], &[]),
    ]);

    let adaptor = CustomGateAdaptor::new(custom); // Decomposes to H, CX

    let result = ToolBuilder::new()
        .with_program(circuit)
        .with_simulator(SparseStab::new(2))
        .with_gate_registry(registry)
        .with_adaptor(adaptor)
        .build();

    assert!(result.is_ok());
}
```

### 8.2 Noise Model Coverage Validation
```rust
#[test]
fn test_build_warns_unhandled_by_noise() {
    let mut registry = GateRegistry::new();
    let custom = registry.register(GateSpec {
        name: "CustomGate",
        quantum_arity: 2,
        category: GateCategory::Custom(1), // Custom category
        ..default()
    });

    let circuit = Circuit::from_gates(vec![
        Gate::custom(custom, &[0, 1], &[]),
    ]);

    let noise = CompositeNoiseModelBuilder::new()
        .with_p1(0.01)  // Only handles SingleQubitUnitary
        .build();

    let result = ToolBuilder::new()
        .with_program(circuit)
        .with_simulator(SparseStab::new(2))
        .with_gate_registry(registry)
        .with_adaptor(...)
        .with_noise(noise)
        .build();

    assert!(matches!(result, Err(ValidationError::UnhandledByNoise(_))));
}
```

## 9. Arity-Based Noise Filtering Tests

```rust
#[test]
fn test_single_qubit_filter_matches_h() {
    let registry = GateRegistry::new();
    let filter = CompositeEventFilter::SingleQubitGate;

    let gate = Gate::h(&[0]);
    assert!(filter.matches(&gate, &registry));
}

#[test]
fn test_single_qubit_filter_rejects_cx() {
    let registry = GateRegistry::new();
    let filter = CompositeEventFilter::SingleQubitGate;

    let gate = Gate::cx(&[0, 1]);
    assert!(!filter.matches(&gate, &registry));
}

#[test]
fn test_two_qubit_filter_matches_custom_2q_gate() {
    let mut registry = GateRegistry::new();
    let custom = registry.register(GateSpec {
        name: "My2QGate",
        quantum_arity: 2,
        ..default()
    });

    let filter = CompositeEventFilter::TwoQubitGate;
    let gate = Gate::custom(custom, &[0, 1], &[]);

    assert!(filter.matches(&gate, &registry));
}

#[test]
fn test_parameterized_filter_matches_rz() {
    let registry = GateRegistry::new();
    let filter = CompositeEventFilter::ParameterizedGate;

    let gate = Gate::rz(Angle64::QUARTER_TURN, &[0]);
    assert!(filter.matches(&gate, &registry));
}

#[test]
fn test_parameterized_filter_rejects_h() {
    let registry = GateRegistry::new();
    let filter = CompositeEventFilter::ParameterizedGate;

    let gate = Gate::h(&[0]);
    assert!(!filter.matches(&gate, &registry));
}

#[test]
fn test_compound_filter() {
    let registry = GateRegistry::new();

    // Two-qubit AND parameterized
    let filter = CompositeEventFilter::And(
        Box::new(CompositeEventFilter::TwoQubitGate),
        Box::new(CompositeEventFilter::ParameterizedGate),
    );

    let rzz = Gate::rzz(Angle64::QUARTER_TURN, &[0, 1]);
    let cx = Gate::cx(&[0, 1]);
    let rz = Gate::rz(Angle64::QUARTER_TURN, &[0]);

    assert!(filter.matches(&rzz, &registry));  // 2Q + parameterized
    assert!(!filter.matches(&cx, &registry));  // 2Q but not parameterized
    assert!(!filter.matches(&rz, &registry));  // parameterized but 1Q
}
```

## 10. Circuit Header / Program Tests

```rust
#[test]
fn test_program_with_custom_gate_header() {
    let program = Program {
        custom_gates: vec![
            GateSpec {
                name: "MyGate",
                quantum_arity: 2,
                angle_arity: 1,
                ..default()
            },
        ],
        gates: vec![
            Gate::new(GateId(256), &[0, 1], &[Angle64::QUARTER_TURN]),
        ],
    };

    let mut registry = GateRegistry::new();
    let ids = program.register_custom_gates(&mut registry);

    assert_eq!(ids.len(), 1);
    assert_eq!(ids[0], GateId(256));
    assert_eq!(registry.get(ids[0]).unwrap().name, "MyGate");
}

#[test]
fn test_program_serialization_roundtrip() {
    let program = Program {
        custom_gates: vec![
            GateSpec { name: "CustomRot", quantum_arity: 2, angle_arity: 3, ..default() },
        ],
        gates: vec![
            Gate::h(&[0]),
            Gate::new(GateId(256), &[0, 1], &[Angle64::QUARTER_TURN, Angle64::HALF_TURN, Angle64::ZERO]),
        ],
    };

    let bytes = program.serialize();
    let loaded = Program::deserialize(&bytes).unwrap();

    assert_eq!(loaded.custom_gates.len(), 1);
    assert_eq!(loaded.custom_gates[0].name, "CustomRot");
    assert_eq!(loaded.gates.len(), 2);
}
```

## 11. Gate Adaptor Tests

```rust
#[test]
fn test_standard_adaptor_decomposes_t() {
    let adaptor = StandardAdaptor::clifford_rz();

    assert!(adaptor.can_adapt(gates::T));

    let decomposed = adaptor.adapt(gates::T, &[QubitId(0)], &[], &[]);

    // T = RZ(π/4)
    assert_eq!(decomposed.len(), 1);
    assert_eq!(decomposed[0].gate_id, gates::RZ);
    assert_eq!(decomposed[0].angles[0], Angle64::HALF_TURN / 4);
}

#[test]
fn test_standard_adaptor_decomposes_swap() {
    let adaptor = StandardAdaptor::clifford_rz();

    let decomposed = adaptor.adapt(gates::SWAP, &[QubitId(0), QubitId(1)], &[], &[]);

    // SWAP = CX CX CX (or similar)
    assert!(decomposed.len() >= 3);
    assert!(decomposed.iter().all(|g| g.gate_id == gates::CX));
}

#[test]
fn test_adaptor_bitset_lookup() {
    let adaptor = StandardAdaptor::clifford_rz();

    // Fast bit test
    assert!(adaptor.can_adapt(gates::T));
    assert!(adaptor.can_adapt(gates::SWAP));
    assert!(!adaptor.can_adapt(gates::H)); // Natively supported
}
```

## 12. Integration Tests

### 12.1 Full Circuit Execution with Custom Gates
```rust
#[test]
fn test_full_execution_with_custom_gate() {
    let mut registry = GateRegistry::new();
    let custom = registry.register(GateSpec {
        name: "MyRotation",
        quantum_arity: 2,
        angle_arity: 1,
        ..default()
    });

    struct MyRotAdaptor(GateId);
    impl GateAdaptor for MyRotAdaptor {
        fn can_adapt(&self, id: GateId) -> bool { id == self.0 }
        fn adapt(&self, _: GateId, q: &[QubitId], a: &[Angle64], _: &[f64]) -> Vec<Gate> {
            vec![
                Gate::cx(&[q[0], q[1]]),
                Gate::rz(a[0], &[q[1]]),
                Gate::cx(&[q[0], q[1]]),
            ]
        }
    }

    let circuit = Circuit::from_gates(vec![
        Gate::prep(&[0, 1]),
        Gate::h(&[0]),
        Gate::new(custom, &[0, 1], &[Angle64::QUARTER_TURN]),
        Gate::measure(&[0, 1]),
    ]);

    let mut tool = ToolBuilder::new()
        .with_program(circuit)
        .with_simulator(SparseStab::new(2))
        .with_gate_registry(registry)
        .with_adaptor(MyRotAdaptor(custom))
        .build()
        .unwrap();

    let outcomes = tool.run();
    assert!(outcomes.get_bit(QubitId(0)).is_some());
}
```

### 12.2 Custom Gate with Noise
```rust
#[test]
fn test_custom_gate_gets_two_qubit_noise() {
    let mut registry = GateRegistry::new();
    let custom = registry.register(GateSpec {
        name: "My2Q",
        quantum_arity: 2,
        category: GateCategory::TwoQubitUnitary,
        ..default()
    });

    let noise = CompositeNoiseModelBuilder::new()
        .with_p2(1.0) // 100% error on 2Q gates for testing
        .build();

    // ... setup adaptor and builder ...

    // Noise should apply to custom 2Q gate via category matching
}
```

## 13. Performance Tests

```rust
#[bench]
fn bench_gate_lookup_core(b: &mut Bencher) {
    let registry = GateRegistry::new();

    b.iter(|| {
        black_box(registry.get(gates::H));
    });
}

#[bench]
fn bench_gate_lookup_user(b: &mut Bencher) {
    let mut registry = GateRegistry::new();
    let id = registry.register(GateSpec { name: "Custom", ..default() });

    b.iter(|| {
        black_box(registry.get(id));
    });
}

#[bench]
fn bench_bitset_contains(b: &mut Bencher) {
    let mut set = GateSupportSet::new();
    for i in 0..100 {
        set.add(GateId(i));
    }

    b.iter(|| {
        black_box(set.supports(GateId(50)));
    });
}

#[bench]
fn bench_canonicalization(b: &mut Bencher) {
    let canon = GateCanonicalizer::standard();

    b.iter(|| {
        black_box(canon.canonicalize(gates::RZ, &[Angle64::QUARTER_TURN]));
    });
}
```

## Summary: Test Categories

| Category | Tests | Priority |
|----------|-------|----------|
| GateId/GateSpec basics | 5 | High |
| GateRegistry | 8 | High |
| Const tables | 3 | High |
| BitVec operations | 5 | High |
| Canonicalization | 6 | High |
| Angle snapping | 5 | Medium |
| Circuit validators | 6 | High |
| Build validation | 5 | High |
| Arity-based filters | 7 | Medium |
| Program headers | 2 | Medium |
| Adaptors | 3 | High |
| Integration | 3 | High |
| Performance | 4 | Medium |

**Total: ~62 tests**
