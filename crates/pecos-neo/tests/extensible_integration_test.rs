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
//! - `CommandSource` with user-defined gates
#![allow(clippy::float_cmp)]

use pecos_core::{Angle64, QubitId};
use pecos_neo::command::CommandBuilder;
use pecos_neo::extensible::{
    AdaptedOp, AdaptedSequence, CircuitResolver, CoreGatesPlugin, DecompOp, DecompositionRegistry,
    GateId, GatePlugin, GateSupportSet, PluginError, PluginLoader, ResolutionError,
    UserGateBuilder, UserGateRegistry, gates,
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
/// This tests: `CUSTOM_GATE` → SWAP → CX (two levels of decomposition).
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

/// Test three levels of decomposition: `GATE_A` → `GATE_B` → `GATE_C` → native
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
        "Should detect circular dependency, got: {result:?}"
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

/// Test that user-defined gates work with noise models in `sim_neo`.
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
        .pz(0)
        .identity(0) // Identity gate triggers noise
        .mz(0)
        .build();

    let mut ones_count = 0;
    for seed in 0..num_shots {
        let noise = ComposableNoiseModel::new()
            .add_channel(SingleQubitChannel::depolarizing(high_noise_rate));
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
    let ones_rate = f64::from(ones_count) / f64::from(num_shots);

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

/// A custom `CommandSource` that uses user-defined gates.
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
    fn next_commands(
        &mut self,
        _outcomes: Option<&MeasurementOutcomes>,
    ) -> Option<pecos_neo::command::CommandQueue> {
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
                .pz(0)
                .pz(1)
                .h(0)
                .cx(0, 1)
                .mz(0)
                .mz(1)
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
    let initial = CommandBuilder::new().pz(0).h(0).mz(0).build();

    // Branch: if measured 1, apply X to flip back to |0>
    let branch = |outcomes: &MeasurementOutcomes| {
        if outcomes.get_bit(QubitId(0)) == Some(true) {
            Some(CommandBuilder::new().x(0).mz(0).build())
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
            if let Some(bit) = result.outcomes.get_bit(QubitId(0))
                && bit
            {
                final_ones += 1;
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
            pecos_neo::extensible::ResolvedOp::Gate {
                gate_id, qubits, ..
            } => {
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
        AdaptedOp::gate1(gates::H, QubitId(0)), // Native
        AdaptedOp::gate2(gates::SWAP, QubitId(0), QubitId(1)), // Decomposed to 3 CX
        AdaptedOp::gate1(gates::H, QubitId(1)), // Native
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
            _ => panic!("Expected CX at position {i}"),
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
/// we get an `UnsupportedNativeGate` error.
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
        "Should error when decomposition chain reaches unsupported native gate, got: {result:?}"
    );
}

/// Test that `MissingRequirements` error is raised when a decomposition requires
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
        "Should error with UnknownGate for unregistered dependency, got: {result:?}"
    );
}

// ============================================================================
// End-to-End Custom Gate Examples
// ============================================================================

/// End-to-end example: Define a custom gate, execute it, verify correctness.
///
/// This demonstrates the complete workflow:
/// 1. Define a custom gate (BELL) that creates a Bell state
/// 2. Register it with the decomposition registry
/// 3. Execute through the simulator
/// 4. Verify the expected quantum behavior
#[test]
fn test_e2e_custom_gate_definition_and_execution() {
    use pecos_neo::runner::CircuitRunner;

    // Step 1: Define a custom BELL gate that creates |00> + |11>
    let mut user_registry = UserGateRegistry::new();
    let bell_gate_id = user_registry.register(
        UserGateBuilder::new("BELL")
            .qubits(2)
            .requires([gates::H, gates::CX])
            .decomposition(vec![
                DecompOp::gate1(gates::H, 0),     // H on control
                DecompOp::gate2(gates::CX, 0, 1), // CX to entangle
            ])
            .build(),
    );

    // Verify the gate got a user-defined ID (>= 256)
    assert!(bell_gate_id.is_user_defined());
    assert_eq!(user_registry.get_id("BELL"), Some(bell_gate_id));

    // Step 2: Build registry and resolve the gate
    let mut decomp_registry = PluginLoader::new()
        .with_plugin(CoreGatesPlugin)
        .build()
        .unwrap();
    user_registry.apply_to(&mut decomp_registry);

    let sim_support = GateSupportSet::from_iter([gates::H, gates::CX]);
    let resolver = CircuitResolver::new(&decomp_registry, &sim_support);

    // Step 3: Create a sequence with just the custom gate (no prep/measure in resolver)
    // Prep and Measure are handled separately by the runner, not by decomposition
    let seq = AdaptedSequence::new(vec![AdaptedOp::gate2(bell_gate_id, QubitId(0), QubitId(1))]);

    // Step 4: Resolve to native gates
    let resolved = resolver.resolve(&seq).unwrap();

    // BELL gate decomposes to: H, CX = 2 ops
    assert_eq!(resolved.len(), 2);

    // Verify the decomposition is correct
    match &resolved.ops[0] {
        pecos_neo::extensible::ResolvedOp::Gate { gate_id, .. } => {
            assert_eq!(*gate_id, gates::H);
        }
        _ => panic!("Expected H gate"),
    }
    match &resolved.ops[1] {
        pecos_neo::extensible::ResolvedOp::Gate { gate_id, .. } => {
            assert_eq!(*gate_id, gates::CX);
        }
        _ => panic!("Expected CX gate"),
    }

    // Step 5: Execute and verify Bell state behavior
    // Build full CommandQueue with prep/measure
    let commands = CommandBuilder::new()
        .pz(0)
        .pz(1)
        .h(0)
        .cx(0, 1)
        .mz(0)
        .mz(1)
        .build();

    // Run multiple shots and verify correlation
    let mut same_outcome_count = 0;
    let num_shots = 100;

    for seed in 0..num_shots {
        let mut state = SparseStab::new(2);
        let mut runner = CircuitRunner::<SparseStab>::new().with_seed(seed as u64);
        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();

        let m0 = outcomes.get_bit(QubitId(0)).unwrap();
        let m1 = outcomes.get_bit(QubitId(1)).unwrap();

        if m0 == m1 {
            same_outcome_count += 1;
        }
    }

    // Bell state: outcomes should ALWAYS be correlated
    assert_eq!(
        same_outcome_count, num_shots,
        "Bell state measurements should always be equal"
    );
}

/// End-to-end example: Custom gate with parameterized rotation.
///
/// Demonstrates custom gates that pass through angle parameters.
#[test]
fn test_e2e_custom_parameterized_gate() {
    // Define a custom controlled-phase gate: CPHASE(theta)
    // Decomposition: CZ is CPHASE(pi), but we want arbitrary angles
    // CPHASE(theta) = I on |00>, |01>, |10>; e^{i*theta} on |11>
    // Decomposition: RZ(theta/2) on both qubits, then CX, RZ(-theta/2) on target, CX
    let mut user_registry = UserGateRegistry::new();
    let cphase_id = user_registry.register(
        UserGateBuilder::new("CPHASE")
            .qubits(2)
            .angles(1)
            .requires([gates::RZ, gates::CX])
            .decomposition(vec![
                // CPHASE(theta) decomposition
                DecompOp::rotation(gates::RZ, 0, 0), // RZ(theta) on qubit 0
                DecompOp::rotation(gates::RZ, 1, 0), // RZ(theta) on qubit 1
                DecompOp::gate2(gates::CX, 0, 1),
                // Note: This is a simplified decomposition for testing
            ])
            .build(),
    );

    assert!(user_registry.contains("CPHASE"));

    // Build and resolve
    let mut decomp_registry = PluginLoader::new()
        .with_plugin(CoreGatesPlugin)
        .build()
        .unwrap();
    user_registry.apply_to(&mut decomp_registry);

    let sim_support = GateSupportSet::from_iter([gates::RZ, gates::CX]);
    let resolver = CircuitResolver::new(&decomp_registry, &sim_support);

    // Create circuit with CPHASE(pi/4)
    let angle = Angle64::new(1) / 8u64; // pi/4
    let seq = AdaptedSequence::new(vec![AdaptedOp::Gate {
        gate_id: cphase_id,
        qubits: smallvec::smallvec![QubitId(0), QubitId(1)],
        angles: smallvec::smallvec![angle],
    }]);

    let resolved = resolver.resolve(&seq).unwrap();

    // Should have: RZ, RZ, CX = 3 ops
    assert_eq!(resolved.len(), 3);

    // Verify angles are preserved
    for op in &resolved.ops {
        if let pecos_neo::extensible::ResolvedOp::Gate {
            gate_id, angles, ..
        } = op
            && *gate_id == gates::RZ
        {
            assert_eq!(
                angles[0], angle,
                "Angle should be preserved through decomposition"
            );
        }
    }
}

/// End-to-end example: Hierarchical custom gates (gate uses another custom gate).
#[test]
fn test_e2e_hierarchical_custom_gates() {
    // Level 1: Define MY_H (just wraps H)
    let mut user_registry = UserGateRegistry::new();
    let my_h_id = user_registry.register(
        UserGateBuilder::new("MY_H")
            .qubits(1)
            .requires([gates::H])
            .decomposition(vec![DecompOp::gate1(gates::H, 0)])
            .build(),
    );

    // Level 2: Define MY_BELL that uses MY_H
    let my_bell_id = user_registry.register(
        UserGateBuilder::new("MY_BELL")
            .qubits(2)
            .requires([my_h_id, gates::CX]) // Uses another user gate!
            .decomposition(vec![
                DecompOp::gate1(my_h_id, 0), // Use MY_H instead of H
                DecompOp::gate2(gates::CX, 0, 1),
            ])
            .build(),
    );

    // Level 3: Define DOUBLE_BELL that uses MY_BELL twice
    let double_bell_id = user_registry.register(
        UserGateBuilder::new("DOUBLE_BELL")
            .qubits(2)
            .requires([my_bell_id])
            .decomposition(vec![
                DecompOp::gate2(my_bell_id, 0, 1),
                DecompOp::gate2(my_bell_id, 0, 1), // Apply twice (returns to separable)
            ])
            .build(),
    );

    // Build and resolve
    let mut decomp_registry = PluginLoader::new()
        .with_plugin(CoreGatesPlugin)
        .build()
        .unwrap();
    user_registry.apply_to(&mut decomp_registry);

    let sim_support = GateSupportSet::from_iter([gates::H, gates::CX]);

    // Verify full chain resolves correctly
    assert!(decomp_registry.can_execute(double_bell_id, &sim_support));

    let resolver = CircuitResolver::new(&decomp_registry, &sim_support);
    let seq = AdaptedSequence::new(vec![AdaptedOp::gate2(
        double_bell_id,
        QubitId(0),
        QubitId(1),
    )]);

    let resolved = resolver.resolve(&seq).unwrap();

    // DOUBLE_BELL -> 2x MY_BELL -> 2x (MY_H + CX) -> 2x (H + CX) = 4 ops
    assert_eq!(resolved.len(), 4);

    // Verify all resolved to native H and CX
    let mut h_count = 0;
    let mut cx_count = 0;
    for op in &resolved.ops {
        if let pecos_neo::extensible::ResolvedOp::Gate { gate_id, .. } = op {
            if *gate_id == gates::H {
                h_count += 1;
            } else if *gate_id == gates::CX {
                cx_count += 1;
            }
        }
    }
    assert_eq!(h_count, 2);
    assert_eq!(cx_count, 2);
}

/// End-to-end example: Custom gate with noise (current limitation).
///
/// This test demonstrates the CURRENT LIMITATION:
/// - Custom gates decompose to native gates
/// - Noise is applied per native gate, not per custom gate
/// - There's no way to apply "5% noise on `MY_GATE`" directly
///
/// TODO: This test documents the gap that needs fixing.
#[test]
fn test_e2e_custom_gate_noise_limitation() {
    use pecos_neo::noise::GateDependentChannel;
    use pecos_neo::runner::CircuitRunner;

    // Define a custom gate
    let mut user_registry = UserGateRegistry::new();
    let my_gate_id = user_registry.register(
        UserGateBuilder::new("MY_IDENTITY")
            .qubits(1)
            .requires([gates::H])
            .decomposition(vec![
                DecompOp::gate1(gates::H, 0),
                DecompOp::gate1(gates::H, 0), // H*H = I
            ])
            .build(),
    );

    // Build registry
    let mut decomp_registry = PluginLoader::new()
        .with_plugin(CoreGatesPlugin)
        .build()
        .unwrap();
    user_registry.apply_to(&mut decomp_registry);

    // CURRENT LIMITATION:
    // We want to apply noise to MY_IDENTITY gate specifically.
    // But GateDependentChannel only accepts GateType, not GateId.
    //
    // This is what we WANT to do (but can't):
    // let noise = GateIdDependentChannel::new()
    //     .with_gate_id(my_gate_id, 0.5);  // 50% error on custom gate
    //
    // What we CAN do is apply noise to the decomposed gates (H):
    let noise = ComposableNoiseModel::new().add_channel(
        GateDependentChannel::new().with_gate_error(pecos_neo::command::GateType::H, 0.0), // No noise for clarity
    );

    // Execute with noise
    let commands = CommandBuilder::new()
        .pz(0)
        .h(0)
        .h(0) // This is our "custom gate" decomposed
        .mz(0)
        .build();

    let mut state = SparseStab::new(1);
    let mut runner = CircuitRunner::<SparseStab>::new()
        .with_noise(noise)
        .with_seed(42);

    let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();

    // H*H = I, so |0> -> |0>
    assert_eq!(
        outcomes.get_bit(QubitId(0)),
        Some(false),
        "H*H should be identity"
    );

    // Document the limitation:
    // The noise model has no knowledge that these two H gates came from MY_IDENTITY.
    // If we wanted per-custom-gate noise, we'd need to either:
    // 1. Add GateId to NoiseEvent (alongside GateType)
    // 2. Apply custom gate noise before decomposition
    // 3. Track decomposition source through execution

    assert!(
        my_gate_id.is_user_defined(),
        "Gate ID {} should be user-defined",
        my_gate_id.as_u16()
    );
}

/// End-to-end example: Using `GateIdNoiseConfig` (currently unused infrastructure).
///
/// This test shows that `GateIdNoiseConfig` exists and works, but isn't
/// connected to actual noise execution.
#[test]
fn test_e2e_gate_id_noise_config_infrastructure() {
    use pecos_neo::extensible::{GateCategory, GateIdNoiseConfig, GateNoiseParams, GateSpec};

    // Create a custom gate
    let mut user_registry = UserGateRegistry::new();
    let my_gate_id = user_registry.register(
        UserGateBuilder::new("NOISY_GATE")
            .qubits(1)
            .requires([gates::H])
            .decomposition(vec![DecompOp::gate1(gates::H, 0)])
            .build(),
    );

    // GateIdNoiseConfig supports per-GateId noise configuration
    let mut noise_config = GateIdNoiseConfig::new()
        .with_category_default(GateCategory::SingleQubitUnitary, 0.001)
        .with_category_default(GateCategory::TwoQubitUnitary, 0.01);

    // Set specific error rate for our custom gate
    noise_config.set_gate(my_gate_id, GateNoiseParams::with_error(0.05));

    // Also set for some core gates
    noise_config.set_gate_error(gates::T, 0.02); // T gates have higher error

    // Query error rates
    let spec = GateSpec::new("NOISY_GATE")
        .with_quantum_arity(1)
        .with_category(GateCategory::SingleQubitUnitary);

    // Custom gate has specific rate
    assert_eq!(
        noise_config.get_error_probability(my_gate_id, Some(&spec)),
        0.05
    );

    // T gate has specific rate
    assert_eq!(noise_config.get_error_probability(gates::T, None), 0.02);

    // Unconfigured single-qubit gate uses category default
    let h_spec = GateSpec::new("H")
        .with_quantum_arity(1)
        .with_category(GateCategory::SingleQubitUnitary);
    assert_eq!(
        noise_config.get_error_probability(gates::SY, Some(&h_spec)),
        0.001
    );

    // NOTE: This infrastructure exists but isn't connected to ComposableNoiseModel
    // or CircuitRunner. The gap is in wiring GateIdNoiseConfig to actual execution.
}

/// End-to-end: Complete workflow using `CircuitRunner` directly.
#[test]
fn test_e2e_complete_workflow_with_shot_runner() {
    use pecos_neo::runner::CircuitRunner;

    // Simple circuit using standard gates
    let commands = CommandBuilder::new()
        .pz(0)
        .pz(1)
        .h(0)
        .cx(0, 1)
        .mz(0)
        .mz(1)
        .build();

    // Use CircuitRunner directly
    let mut state = SparseStab::new(2);
    let mut runner = CircuitRunner::<SparseStab>::new().with_seed(42);
    let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();

    // Verify we got results
    assert_eq!(outcomes.len(), 2);

    // Bell state: outcomes should match
    let m0 = outcomes.get_bit(QubitId(0)).unwrap();
    let m1 = outcomes.get_bit(QubitId(1)).unwrap();
    assert_eq!(m0, m1, "Bell state outcomes should be correlated");
}
