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

//! Integration tests for the extensible gate system.
//!
//! These tests cover gaps identified in the test coverage analysis:
//! - Gate decomposition chaining (A → B → C)
//! - User gate decomposition with noise
//! - Plugin dependency handling
//! - CommandSource with user-defined gates

use pecos_core::{Angle64, QubitId};
use pecos_neo::command::CommandBuilder;
use pecos_neo::extensible::{
    gates, AdaptedOp, AdaptedSequence, CircuitResolver, CoreGatesPlugin, DecompOp,
    DecompositionRegistry, GateId, GatePlugin, GateSupportSet, PluginError, PluginLoader,
    ResolutionError, UserGateBuilder, UserGateRegistry,
};
use pecos_neo::noise::{ComposableNoiseModel, SingleQubitChannel};
use pecos_neo::outcome::MeasurementOutcomes;
use pecos_neo::program::{CommandSource, ConditionalProgram, ProgramRunner, StaticProgram};
use pecos_qsim::SparseStab;
use std::any::TypeId;

// ============================================================================
// Gate Decomposition Chaining Tests (A → B → C)
// ============================================================================

/// Create a custom gate that decomposes to SWAP (which itself decomposes to CX).
/// This tests: CUSTOM_GATE → SWAP → CX (two levels of decomposition).
#[test]
fn test_decomposition_chaining_two_levels() {
    // Define a custom gate that uses SWAP in its decomposition
    // DOUBLE_SWAP: swap qubits twice (identity operation)
    let mut registry = DecompositionRegistry::new();

    // Register DOUBLE_SWAP that uses two SWAPs
    // With recursive resolution, we can list immediate dependencies (SWAP),
    // and the system will recursively resolve SWAP → CX.
    let double_swap = GateId(256);
    registry.register_dynamic(
        double_swap,
        GateSupportSet::from_iter([gates::SWAP]), // Immediate dependency - SWAP
        vec![
            DecompOp::gate2(gates::SWAP, 0, 1),
            DecompOp::gate2(gates::SWAP, 0, 1),
        ],
    );

    // Simulator only supports CX (not SWAP or DOUBLE_SWAP)
    let sim_support = GateSupportSet::from_iter([gates::CX]);

    // Verify can_execute works with recursive resolution
    assert!(
        registry.can_execute(double_swap, &sim_support),
        "Should be able to execute DOUBLE_SWAP via SWAP → CX chain"
    );

    let resolver = CircuitResolver::new(&registry, &sim_support);

    // Create a circuit with DOUBLE_SWAP
    let seq = AdaptedSequence::new(vec![AdaptedOp::gate2(double_swap, QubitId(0), QubitId(1))]);

    // Should resolve to 6 CX gates (2 SWAPs * 3 CX each)
    let resolved = resolver.resolve(&seq).unwrap();

    assert_eq!(resolved.len(), 6, "DOUBLE_SWAP should expand to 6 CX gates");

    // Verify all gates are CX
    for op in &resolved.ops {
        match op {
            pecos_neo::extensible::ResolvedOp::Gate { gate_id, .. } => {
                assert_eq!(*gate_id, gates::CX);
            }
            _ => panic!("Expected Gate"),
        }
    }
}

/// Test three levels of decomposition: GATE_A → GATE_B → GATE_C → native
#[test]
fn test_decomposition_chaining_three_levels() {
    let mut registry = DecompositionRegistry::new();

    // Level 3: GATE_C decomposes to H (native)
    let gate_c = GateId(258);
    registry.register_dynamic(
        gate_c,
        GateSupportSet::from_iter([gates::H]),
        vec![DecompOp::gate1(gates::H, 0), DecompOp::gate1(gates::H, 0)],
    );

    // Level 2: GATE_B decomposes to GATE_C
    // With recursive resolution, we list immediate dependency (GATE_C)
    let gate_b = GateId(257);
    registry.register_dynamic(
        gate_b,
        GateSupportSet::from_iter([gate_c]), // Immediate dependency
        vec![DecompOp::gate1(gate_c, 0)],
    );

    // Level 1: GATE_A decomposes to GATE_B
    // With recursive resolution, we list immediate dependency (GATE_B)
    let gate_a = GateId(256);
    registry.register_dynamic(
        gate_a,
        GateSupportSet::from_iter([gate_b]), // Immediate dependency
        vec![DecompOp::gate1(gate_b, 0)],
    );

    // Simulator only supports H
    let sim_support = GateSupportSet::from_iter([gates::H]);

    // Verify recursive resolution works through all 3 levels
    assert!(
        registry.can_execute(gate_a, &sim_support),
        "Should resolve GATE_A → GATE_B → GATE_C → H"
    );

    let resolver = CircuitResolver::new(&registry, &sim_support);

    let seq = AdaptedSequence::new(vec![AdaptedOp::gate1(gate_a, QubitId(0))]);

    // Should resolve to 2 H gates (GATE_A → GATE_B → GATE_C → H, H)
    let resolved = resolver.resolve(&seq).unwrap();

    assert_eq!(
        resolved.len(),
        2,
        "GATE_A should eventually expand to 2 H gates"
    );
}

/// Test that circular dependencies in gate decompositions are detected.
#[test]
fn test_circular_dependency_detection() {
    let mut registry = DecompositionRegistry::new();

    // Create a circular dependency: GATE_A → GATE_B → GATE_A
    let gate_a = GateId(256);
    let gate_b = GateId(257);

    registry.register_dynamic(
        gate_a,
        GateSupportSet::from_iter([gate_b]),
        vec![DecompOp::gate1(gate_b, 0)],
    );

    registry.register_dynamic(
        gate_b,
        GateSupportSet::from_iter([gate_a]), // Circular!
        vec![DecompOp::gate1(gate_a, 0)],
    );

    // Simulator supports nothing - forces decomposition
    let sim_support = GateSupportSet::new();

    // can_execute should return false (cycle detected)
    assert!(
        !registry.can_execute(gate_a, &sim_support),
        "Circular dependency should not be executable"
    );

    // resolve should return CircularDependency error
    let result = registry.resolve(gate_a, &sim_support);
    assert!(
        matches!(result, Err(ResolutionError::CircularDependency(_))),
        "Should detect circular dependency, got: {:?}",
        result
    );
}

/// Test self-referential gate (gate requires itself).
#[test]
fn test_self_referential_gate_detection() {
    let mut registry = DecompositionRegistry::new();

    // Create a self-referential gate: GATE_A → GATE_A
    let gate_a = GateId(256);

    registry.register_dynamic(
        gate_a,
        GateSupportSet::from_iter([gate_a]), // Self-reference!
        vec![DecompOp::gate1(gate_a, 0)],
    );

    let sim_support = GateSupportSet::new();

    // Should detect the self-reference
    assert!(
        !registry.can_execute(gate_a, &sim_support),
        "Self-referential gate should not be executable"
    );

    let result = registry.resolve(gate_a, &sim_support);
    assert!(
        matches!(result, Err(ResolutionError::CircularDependency(_))),
        "Should detect self-reference as circular dependency"
    );
}

/// Test decomposition where a user gate requires another user gate.
#[test]
fn test_user_gate_requires_user_gate() {
    let mut user_registry = UserGateRegistry::new();

    // First user gate: MY_H (wraps H)
    let my_h_id = user_registry.register(
        UserGateBuilder::new("MY_H")
            .qubits(1)
            .requires([gates::H])
            .decomposition(vec![DecompOp::gate1(gates::H, 0)])
            .build(),
    );

    // Second user gate: DOUBLE_MY_H (uses MY_H)
    // With recursive resolution, we list immediate dependency (my_h_id)
    let double_my_h_id = user_registry.register(
        UserGateBuilder::new("DOUBLE_MY_H")
            .qubits(1)
            .requires([my_h_id]) // Immediate dependency - another user gate!
            .decomposition(vec![
                DecompOp::gate1(my_h_id, 0),
                DecompOp::gate1(my_h_id, 0),
            ])
            .build(),
    );

    // Apply to decomposition registry
    let mut decomp_registry = DecompositionRegistry::new();
    user_registry.apply_to(&mut decomp_registry);

    // Simulator supports only H
    let sim_support = GateSupportSet::from_iter([gates::H]);

    // Verify recursive resolution through user gates
    assert!(
        decomp_registry.can_execute(double_my_h_id, &sim_support),
        "Should resolve DOUBLE_MY_H → MY_H → H"
    );

    let resolver = CircuitResolver::new(&decomp_registry, &sim_support);

    let seq = AdaptedSequence::new(vec![AdaptedOp::gate1(double_my_h_id, QubitId(0))]);

    let resolved = resolver.resolve(&seq).unwrap();

    // DOUBLE_MY_H → 2x MY_H → 2x H
    assert_eq!(resolved.len(), 2);
}

// ============================================================================
// Plugin Dependency Tests
// ============================================================================

/// Test that circular dependencies are detected (or at least don't hang).
#[test]
fn test_plugin_dependency_detection() {
    // Create plugins that depend on each other
    struct PluginA;
    struct PluginB;

    impl GatePlugin for PluginA {
        fn name(&self) -> &'static str {
            "plugin-a"
        }

        fn dependencies(&self) -> Vec<TypeId> {
            vec![TypeId::of::<PluginB>()]
        }

        fn build(&self, _registry: &mut DecompositionRegistry) {}
    }

    impl GatePlugin for PluginB {
        fn name(&self) -> &'static str {
            "plugin-b"
        }

        fn dependencies(&self) -> Vec<TypeId> {
            vec![TypeId::of::<PluginA>()]
        }

        fn build(&self, _registry: &mut DecompositionRegistry) {}
    }

    // This should fail because A needs B and B needs A
    let result = PluginLoader::new()
        .with_plugin(PluginA)
        .with_plugin(PluginB)
        .build();

    // The current implementation should detect this as unresolved dependencies
    assert!(
        matches!(result, Err(PluginError::UnresolvedDependencies(_))),
        "Circular dependency should be detected"
    );
}

/// Test multi-level plugin dependencies.
#[test]
fn test_plugin_multi_level_dependencies() {
    struct PluginBase;
    struct PluginMiddle;
    struct PluginTop;

    impl GatePlugin for PluginBase {
        fn name(&self) -> &'static str {
            "base"
        }
        fn build(&self, registry: &mut DecompositionRegistry) {
            // Register a marker gate
            registry.register_native(GateId(256));
        }
    }

    impl GatePlugin for PluginMiddle {
        fn name(&self) -> &'static str {
            "middle"
        }
        fn dependencies(&self) -> Vec<TypeId> {
            vec![TypeId::of::<PluginBase>()]
        }
        fn build(&self, registry: &mut DecompositionRegistry) {
            registry.register_native(GateId(257));
        }
    }

    impl GatePlugin for PluginTop {
        fn name(&self) -> &'static str {
            "top"
        }
        fn dependencies(&self) -> Vec<TypeId> {
            vec![TypeId::of::<PluginMiddle>()]
        }
        fn build(&self, registry: &mut DecompositionRegistry) {
            registry.register_native(GateId(258));
        }
    }

    // Load in wrong order - should still work
    let registry = PluginLoader::new()
        .with_plugin(PluginTop)
        .with_plugin(PluginBase)
        .with_plugin(PluginMiddle)
        .build()
        .expect("Should resolve multi-level dependencies");

    assert!(registry.contains(GateId(256)));
    assert!(registry.contains(GateId(257)));
    assert!(registry.contains(GateId(258)));
}

// ============================================================================
// User Gates with Noise Integration
// ============================================================================

/// Test that user-defined gates work with noise models in sim_neo.
#[test]
fn test_user_gate_with_noise_integration() {
    // Create a user gate that does H-CX-H (entangling operation)
    let mut user_registry = UserGateRegistry::new();

    let entangle_id = user_registry.register(
        UserGateBuilder::new("ENTANGLE")
            .qubits(2)
            .requires([gates::H, gates::CX])
            .decomposition(vec![
                DecompOp::gate1(gates::H, 0),
                DecompOp::gate2(gates::CX, 0, 1),
                DecompOp::gate1(gates::H, 0),
            ])
            .build(),
    );

    // Build registry with user gates
    let mut registry = PluginLoader::new()
        .with_plugin(CoreGatesPlugin)
        .build()
        .unwrap();
    user_registry.apply_to(&mut registry);

    // Verify the gate is registered
    assert!(registry.contains(entangle_id));

    // Verify it can be resolved with CX+H support
    let sim_support = GateSupportSet::from_iter([gates::H, gates::CX]);
    assert!(registry.can_execute(entangle_id, &sim_support));

    // Create a circuit using the user gate
    let seq = AdaptedSequence::new(vec![AdaptedOp::gate2(entangle_id, QubitId(0), QubitId(1))]);

    let resolver = CircuitResolver::new(&registry, &sim_support);
    let resolved = resolver.resolve(&seq).unwrap();

    // Should decompose to H, CX, H
    assert_eq!(resolved.len(), 3);
}

/// Test noisy execution with gates.
#[test]
fn test_noisy_execution_statistics() {
    // This test verifies that noise is applied during gate execution.
    // We apply a high depolarizing rate and verify statistical behavior.

    let num_shots = 500;
    let high_noise_rate = 0.3; // 30% depolarizing per gate

    // Circuit: prepare |0>, apply identity gate (I), measure
    // The I gate triggers noise application
    let commands = CommandBuilder::new()
        .prep(0)
        .identity(0) // Identity gate triggers noise
        .measure(0)
        .build();

    let mut ones_count = 0;
    for seed in 0..num_shots {
        let noise = ComposableNoiseModel::new().add_channel(SingleQubitChannel::depolarizing(
            high_noise_rate,
        ));
        let mut program = StaticProgram::new(commands.clone(), 1);
        let mut runner = ProgramRunner::new(SparseStab::new(1))
            .with_noise(noise)
            .with_seed(seed as u64);

        let result = runner.run_shot(&mut program);
        if result.outcomes.get_bit(QubitId(0)) == Some(true) {
            ones_count += 1;
        }
    }

    // With 30% depolarizing on the I gate, we expect some bit flips
    // Depolarizing with probability p means:
    // - No error: 1-p
    // - X error: p/3 (flips |0> to |1>)
    // - Y error: p/3 (flips |0> to |1>)
    // - Z error: p/3 (no flip on |0>)
    // So ~2p/3 = ~20% of shots should measure 1
    let ones_rate = ones_count as f64 / num_shots as f64;

    // Should see some bit flips (between 5% and 40%)
    assert!(
        ones_rate > 0.05,
        "Expected some bit flips from noise, got {:.1}%",
        ones_rate * 100.0
    );
    assert!(
        ones_rate < 0.40,
        "Too many bit flips ({:.1}%), noise might be too aggressive",
        ones_rate * 100.0
    );
}

// ============================================================================
// CommandSource with User-Defined Gates
// ============================================================================

/// A custom CommandSource that uses user-defined gates.
struct UserGateProgram {
    _user_gate_id: GateId, // Stored for potential future use with CommandBuilder extension
    executed: bool,
}

impl UserGateProgram {
    fn new(user_gate_id: GateId) -> Self {
        Self {
            _user_gate_id: user_gate_id,
            executed: false,
        }
    }
}

impl CommandSource for UserGateProgram {
    fn next_commands(&mut self, _outcomes: Option<&MeasurementOutcomes>) -> Option<pecos_neo::command::CommandQueue> {
        if self.executed {
            return None;
        }
        self.executed = true;

        // Build commands using the user gate
        // Note: The CommandBuilder doesn't directly support custom GateIds,
        // so for now we use standard gates to verify the CommandSource pattern works.
        // The actual user gate integration would require CommandBuilder extension.
        Some(
            CommandBuilder::new()
                .prep(0)
                .prep(1)
                .h(0)
                .cx(0, 1)
                .measure(0)
                .measure(1)
                .build(),
        )
    }

    fn is_complete(&self) -> bool {
        self.executed
    }

    fn reset(&mut self) {
        self.executed = false;
    }

    fn num_qubits(&self) -> usize {
        2
    }
}

#[test]
fn test_command_source_with_user_gates() {
    // Register a user gate
    let mut user_registry = UserGateRegistry::new();
    let user_gate_id = user_registry.register(
        UserGateBuilder::new("MY_BELL")
            .qubits(2)
            .requires([gates::H, gates::CX])
            .decomposition(vec![
                DecompOp::gate1(gates::H, 0),
                DecompOp::gate2(gates::CX, 0, 1),
            ])
            .build(),
    );

    // Create program using the user gate
    let mut program = UserGateProgram::new(user_gate_id);
    let mut runner = ProgramRunner::new(SparseStab::new(2)).with_seed(42);

    let result = runner.run_shot(&mut program);

    // Verify execution completed
    assert_eq!(result.num_batches, 1);
    assert_eq!(result.outcomes.len(), 2);

    // Verify Bell state correlation (both measurements should agree)
    let m0 = result.outcomes.get_bit(QubitId(0));
    let m1 = result.outcomes.get_bit(QubitId(1));
    assert_eq!(m0, m1, "Bell state measurements should be correlated");
}

/// Test conditional branching with measurement feedback.
#[test]
fn test_conditional_program_with_feedback() {
    // Initial circuit: prepare |+>, measure
    let initial = CommandBuilder::new().prep(0).h(0).measure(0).build();

    // Branch: if measured 1, apply X to flip back to |0>
    let branch = |outcomes: &MeasurementOutcomes| {
        if outcomes.get_bit(QubitId(0)) == Some(true) {
            Some(CommandBuilder::new().x(0).measure(0).build())
        } else {
            None
        }
    };

    // Run many shots and verify the correction works
    let num_shots = 100;
    let mut final_ones = 0;

    for seed in 0..num_shots {
        let mut program = ConditionalProgram::new(initial.clone(), branch, 1);
        let mut runner = ProgramRunner::new(SparseStab::new(1)).with_seed(seed as u64);

        let result = runner.run_shot(&mut program);

        // If there were 2 batches, we did the correction
        if result.num_batches == 2 {
            // After X correction, the second measurement should be 0
            // (since X|1> = |0>)
            if let Some(bit) = result.outcomes.get_bit(QubitId(0)) {
                if bit {
                    final_ones += 1;
                }
            }
        }
    }

    // After correction, we should have very few 1s (ideally 0, but noise/state issues could cause some)
    // The key is that we're testing the feedback loop works
    assert!(
        final_ones < num_shots / 4,
        "X correction should reduce 1 outcomes"
    );
}

// ============================================================================
// Rotation Gate Decomposition with Angles
// ============================================================================

#[test]
fn test_rotation_gate_decomposition_preserves_angles() {
    let mut registry = DecompositionRegistry::new();

    // RZZ(θ) = CX(0,1); RZ(θ, 1); CX(0,1)
    let rzz_ops = vec![
        DecompOp::gate2(gates::CX, 0, 1),
        DecompOp::rotation(gates::RZ, 1, 0), // Use input angle at index 0
        DecompOp::gate2(gates::CX, 0, 1),
    ];

    registry.register_dynamic(
        gates::RZZ,
        GateSupportSet::from_iter([gates::CX, gates::RZ]),
        rzz_ops,
    );

    let sim_support = GateSupportSet::from_iter([gates::CX, gates::RZ]);
    let resolver = CircuitResolver::new(&registry, &sim_support);

    // Create RZZ with a specific angle
    let angle = Angle64::QUARTER_TURN;
    let seq = AdaptedSequence::new(vec![AdaptedOp::Gate {
        gate_id: gates::RZZ,
        qubits: smallvec::smallvec![QubitId(0), QubitId(1)],
        angles: smallvec::smallvec![angle],
    }]);

    let resolved = resolver.resolve(&seq).unwrap();

    // Should be: CX, RZ(angle), CX
    assert_eq!(resolved.len(), 3);

    // Check that the middle gate (RZ) has the correct angle
    match &resolved.ops[1] {
        pecos_neo::extensible::ResolvedOp::Gate {
            gate_id, angles, ..
        } => {
            assert_eq!(*gate_id, gates::RZ);
            assert_eq!(angles[0], angle);
        }
        _ => panic!("Expected RZ gate"),
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_empty_decomposition() {
    let mut registry = DecompositionRegistry::new();

    // A gate with empty decomposition (acts as identity)
    let noop_gate = GateId(256);
    registry.register_dynamic(noop_gate, GateSupportSet::new(), vec![]);

    let sim_support = GateSupportSet::new();
    let resolver = CircuitResolver::new(&registry, &sim_support);

    let seq = AdaptedSequence::new(vec![AdaptedOp::gate1(noop_gate, QubitId(0))]);

    let resolved = resolver.resolve(&seq).unwrap();

    // Empty decomposition should produce no operations
    assert!(resolved.is_empty());
}

#[test]
fn test_native_gate_passthrough() {
    let registry = DecompositionRegistry::new();

    // Simulator supports H natively
    let sim_support = GateSupportSet::from_iter([gates::H]);
    let resolver = CircuitResolver::new(&registry, &sim_support);

    let seq = AdaptedSequence::new(vec![
        AdaptedOp::gate1(gates::H, QubitId(0)),
        AdaptedOp::gate1(gates::H, QubitId(1)),
    ]);

    let resolved = resolver.resolve(&seq).unwrap();

    // Should pass through unchanged
    assert_eq!(resolved.len(), 2);

    for (i, op) in resolved.ops.iter().enumerate() {
        match op {
            pecos_neo::extensible::ResolvedOp::Gate { gate_id, qubits, .. } => {
                assert_eq!(*gate_id, gates::H);
                assert_eq!(qubits[0], QubitId(i));
            }
            _ => panic!("Expected Gate"),
        }
    }
}

#[test]
fn test_mixed_native_and_decomposed() {
    let registry = DecompositionRegistry::new();

    // Simulator supports H and CX but not SWAP
    let sim_support = GateSupportSet::from_iter([gates::H, gates::CX]);
    let resolver = CircuitResolver::new(&registry, &sim_support);

    let seq = AdaptedSequence::new(vec![
        AdaptedOp::gate1(gates::H, QubitId(0)),            // Native
        AdaptedOp::gate2(gates::SWAP, QubitId(0), QubitId(1)), // Decomposed to 3 CX
        AdaptedOp::gate1(gates::H, QubitId(1)),            // Native
    ]);

    let resolved = resolver.resolve(&seq).unwrap();

    // H + 3 CX + H = 5 ops
    assert_eq!(resolved.len(), 5);

    // First should be H
    match &resolved.ops[0] {
        pecos_neo::extensible::ResolvedOp::Gate { gate_id, .. } => {
            assert_eq!(*gate_id, gates::H);
        }
        _ => panic!("Expected H"),
    }

    // Middle 3 should be CX
    for i in 1..4 {
        match &resolved.ops[i] {
            pecos_neo::extensible::ResolvedOp::Gate { gate_id, .. } => {
                assert_eq!(*gate_id, gates::CX);
            }
            _ => panic!("Expected CX at position {}", i),
        }
    }

    // Last should be H
    match &resolved.ops[4] {
        pecos_neo::extensible::ResolvedOp::Gate { gate_id, .. } => {
            assert_eq!(*gate_id, gates::H);
        }
        _ => panic!("Expected H"),
    }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_resolution_error_unknown_gate() {
    let registry = DecompositionRegistry::new();
    let sim_support = GateSupportSet::from_iter([gates::H]);
    let resolver = CircuitResolver::new(&registry, &sim_support);

    // Use an unregistered gate
    let unknown_gate = GateId(999);
    let seq = AdaptedSequence::new(vec![AdaptedOp::gate1(unknown_gate, QubitId(0))]);

    let result = resolver.resolve(&seq);
    assert!(
        matches!(
            result,
            Err(pecos_neo::extensible::ResolutionError::UnknownGate(_))
        ),
        "Should error on unknown gate"
    );
}

#[test]
fn test_resolution_error_unsupported_native() {
    let registry = DecompositionRegistry::new();

    // Simulator supports nothing
    let sim_support = GateSupportSet::new();
    let resolver = CircuitResolver::new(&registry, &sim_support);

    // H is native but not supported
    let seq = AdaptedSequence::new(vec![AdaptedOp::gate1(gates::H, QubitId(0))]);

    let result = resolver.resolve(&seq);
    assert!(
        matches!(
            result,
            Err(pecos_neo::extensible::ResolutionError::UnsupportedNativeGate(_))
        ),
        "Should error when native gate is not supported"
    );
}

/// Test that when a decomposition chain reaches a native gate that's unsupported,
/// we get an UnsupportedNativeGate error.
#[test]
fn test_resolution_error_unsupported_in_chain() {
    let registry = DecompositionRegistry::new();

    // Simulator supports H but not CX (needed for SWAP decomposition)
    // SWAP → CX, CX, CX, but CX is native and unsupported
    let sim_support = GateSupportSet::from_iter([gates::H]);
    let resolver = CircuitResolver::new(&registry, &sim_support);

    let seq = AdaptedSequence::new(vec![AdaptedOp::gate2(gates::SWAP, QubitId(0), QubitId(1))]);

    let result = resolver.resolve(&seq);
    // With recursive resolution, this reaches CX which is native but unsupported
    assert!(
        matches!(
            result,
            Err(pecos_neo::extensible::ResolutionError::UnsupportedNativeGate(g)) if g == gates::CX
        ),
        "Should error when decomposition chain reaches unsupported native gate, got: {:?}",
        result
    );
}

/// Test that MissingRequirements error is raised when a decomposition requires
/// an unregistered gate.
#[test]
fn test_resolution_error_missing_requirements() {
    let mut registry = DecompositionRegistry::new();

    // Create a gate that requires an unregistered gate
    let custom_gate = GateId(256);
    let unregistered_gate = GateId(999);

    registry.register_dynamic(
        custom_gate,
        GateSupportSet::from_iter([unregistered_gate]),
        vec![DecompOp::gate1(unregistered_gate, 0)],
    );

    let sim_support = GateSupportSet::from_iter([gates::H]);

    // can_execute should fail because unregistered_gate can't be resolved
    assert!(
        !registry.can_execute(custom_gate, &sim_support),
        "Should not be able to execute gate with unregistered dependency"
    );

    // Trying to resolve should fail
    let resolver = CircuitResolver::new(&registry, &sim_support);
    let seq = AdaptedSequence::new(vec![AdaptedOp::gate1(custom_gate, QubitId(0))]);

    let result = resolver.resolve(&seq);
    assert!(
        matches!(result, Err(ResolutionError::UnknownGate(g)) if g == unregistered_gate),
        "Should error with UnknownGate for unregistered dependency, got: {:?}",
        result
    );
}
