# ExtendedRunner: GateId-Based Circuit Execution

## Overview

`ExtendedRunner` executes circuits built with `GateId` (via `OpBuilder` / `AdaptedSequence`), providing:

- **Trait-based native execution** - Compile-time checked via `CliffordGateable` / `ArbitraryRotationGateable`
- **Custom gate overrides** - Swap implementations for any gate, including core gates
- **Automatic decomposition** - Fall back to `GateDefinitions` when no native support
- **Unified API** - Single `run()` method, constructor determines capabilities

## ExtendedRunner vs ShotRunner

| Feature | `ShotRunner` | `ExtendedRunner` |
|---------|--------------|------------------|
| Circuit type | `CommandQueue` | `AdaptedSequence` |
| Gate identifier | `GateType` (enum) | `GateId` (u16) |
| Custom gates | Not supported | Full support |
| Gate overrides | Not supported | `GateOverrides` |
| Decomposition | Not supported | Via `GateDefinitions` |

Use `ShotRunner` for simple circuits with core gates only. Use `ExtendedRunner` when you need custom gates, decomposition, or gate overrides.

## Gate Execution Order

When `ExtendedRunner` encounters a gate, it tries execution in this order:

```
1. Overrides     - Check GateOverrides registry
2. Clifford      - Try CliffordGateable trait methods
3. Rotation      - Try ArbitraryRotationGateable (if rotations() was used)
4. Decomposition - Expand via GateDefinitions
5. Error         - ExecutionError::NoDecomposition
```

This is fail-fast: if a gate can't be handled, execution stops with an error.

## Basic Usage

### Clifford-Only Circuits

```rust
use pecos_neo::prelude::*;
use pecos_qsim::SparseStab;

let definitions = GateDefinitions::new();

let circuit = OpBuilder::new()
    .pz(QubitId(0))
    .h(QubitId(0))
    .cx(QubitId(0), QubitId(1))
    .mz(QubitId(0), ResultId(0))
    .mz(QubitId(1), ResultId(1))
    .build();

let mut runner = ExtendedRunner::new(SparseStab::new(2), definitions);
let outcomes = runner.run(&circuit)?;
```

### Circuits with Rotation Gates

```rust
use pecos_neo::prelude::*;
use pecos_qsim::StateVec;

let definitions = GateDefinitions::new();

let circuit = OpBuilder::new()
    .pz(QubitId(0))
    .rx(QubitId(0), Angle64::QUARTER_TURN)  // Rotation gate
    .t(QubitId(0))                           // T gate
    .mz(QubitId(0), ResultId(0))
    .build();

// Use rotations() constructor for native rotation support
let mut runner = ExtendedRunner::rotations(StateVec::new(1), definitions);
let outcomes = runner.run(&circuit)?;  // Same run() method!
```

**Key insight**: The constructor (`new()` vs `rotations()`) determines which gates are native. The `run()` method is the same.

## Gate Overrides

`GateOverrides` lets you provide custom implementations for any gate:

```rust
use pecos_neo::prelude::*;
use pecos_qsim::SparseStab;

// Register custom implementations
let overrides: GateOverrides<SparseStab> = GateOverrides::new()
    // Custom gate implemented as H
    .register(my_custom_gate, |sim, qubits, _angles| {
        sim.h(qubits);
        true
    })
    // Override core H gate (e.g., for debugging)
    .register(gates::H, |sim, qubits, _angles| {
        println!("H gate on {:?}", qubits);
        sim.h(qubits);
        true
    });

let mut runner = ExtendedRunner::new(SparseStab::new(1), definitions)
    .with_overrides(overrides);
```

### Use Cases for Overrides

1. **Custom gates without decomposition** - Provide native implementation
2. **Debugging** - Add logging to specific gates
3. **Testing** - Replace gates with mocks
4. **Performance** - Optimized implementations for specific gates
5. **Temporary swaps** - Test alternative implementations

## Decomposition

When a gate has no override and isn't natively supported, `ExtendedRunner` looks up its decomposition in `GateDefinitions`:

```rust
use pecos_neo::prelude::*;

// Register a custom gate with decomposition
let mut definitions = GateDefinitions::new();
let my_gate = definitions.register(
    GateSpec::new("MyGate")
        .with_quantum_arity(1)
        .with_decomposition(|q, _| vec![
            (gates::H, vec![q[0]], vec![]),
            (gates::SZ, vec![q[0]], vec![]),
            (gates::H, vec![q[0]], vec![]),
        ])
);

let circuit = OpBuilder::new()
    .pz(QubitId(0))
    .gate1(my_gate, QubitId(0))  // Decomposes to H-SZ-H
    .mz(QubitId(0), ResultId(0))
    .build();

let mut runner = ExtendedRunner::new(SparseStab::new(1), definitions);
let outcomes = runner.run(&circuit)?;  // my_gate is expanded automatically
```

### Decomposition Depth

To prevent infinite recursion, decomposition has a maximum depth (default: 10):

```rust
let runner = ExtendedRunner::new(sim, definitions)
    .with_max_decomp_depth(20);  // Increase if needed
```

## Noise Integration

`ExtendedRunner` integrates with `ComposableNoiseModel`:

```rust
use pecos_neo::prelude::*;

let noise = ComposableNoiseModel::new()
    .add_channel(SingleQubitChannel::depolarizing(0.001))
    .add_channel(TwoQubitChannel::depolarizing(0.01));

let mut runner = ExtendedRunner::new(SparseStab::new(2), definitions)
    .with_noise(noise)
    .with_seed(42);

let outcomes = runner.run(&circuit)?;
```

### Noise Events

`ExtendedRunner` emits `NoiseEvent::BeforeGate` and `NoiseEvent::AfterGate` with full gate metadata including `GateId`, enabling noise models to handle custom gates:

```rust
NoiseEvent::AfterGate {
    gate_type: GateType::I,     // Placeholder for custom gates
    qubits: &[QubitId(0)],
    angles: &[],
    gate_id: Some(my_custom_gate),  // Actual gate identity
}
```

## Multiple Shots

```rust
let mut runner = ExtendedRunner::new(sim, definitions)
    .with_seed(42);

// run() keeps state for inspection
let outcomes = runner.run(&circuit)?;
println!("First: {:?}", outcomes);

// run_shot() returns outcomes and resets
for _ in 0..1000 {
    let outcomes = runner.run_shot(&circuit)?;
    // Process each shot...
}
```

## API Reference

### Constructors

| Constructor | Trait Bound | Native Gates |
|-------------|-------------|--------------|
| `new(sim, defs)` | `CliffordGateable` | H, X, Y, Z, SX, SY, SZ, CX, CY, CZ, SWAP, etc. |
| `rotations(sim, defs)` | `+ ArbitraryRotationGateable` | + T, Tdg, RX, RY, RZ, RXX, RYY, RZZ |

### Builder Methods

```rust
runner
    .with_noise(noise)           // Add noise model
    .with_seed(42)               // Set RNG seed
    .with_overrides(overrides)   // Add gate overrides
    .with_max_decomp_depth(20)   // Set decomposition limit
```

### Execution Methods

```rust
runner.run(&circuit)?           // Execute, keep outcomes
runner.run_shot(&circuit)?      // Execute, return outcomes, reset
```

### Inspection

```rust
runner.simulator()              // &S - read-only simulator access
runner.simulator_mut()          // &mut S - mutable simulator access
runner.definitions()            // &GateDefinitions
runner.has_rotation_support()   // bool - was rotations() used?
runner.has_override(gate_id)    // bool - is gate overridden?
```

## Universal Simulation (Rotation Gates via ShotRunner)

For simple circuits using `ShotRunner` with a state vector simulator that supports
arbitrary rotation gates, use `execute_all()` instead of `execute()`:

```rust
use pecos_neo::prelude::*;
use pecos_qsim::StateVec;

let commands = CommandBuilder::new()
    .pz(0)
    .rx(Angle64::HALF_TURN, 0)  // RX(pi) = X gate
    .rzz(Angle64::QUARTER_TURN, 0, 1)  // RZZ(pi/2)
    .mz(0)
    .build();

let mut runner = ShotRunner::new(StateVec::new(2));
let outcomes = runner.execute_all(&commands);
```

Supported rotation gates: RX, RY, RZ, T, Tdg, U, R1XY, RXX, RYY, RZZ, CRZ, CCX (Toffoli).

CCX and CRZ are automatically decomposed into supported gates.

## Error Handling

```rust
pub enum ExecutionError {
    /// Gate has no native support, no override, and no decomposition
    NoDecomposition { gate_id: GateId },

    /// Decomposition chain exceeded max depth (possible infinite loop)
    MaxDecompositionDepthExceeded,
}
```

## Design Rationale

### Why Trait Bounds as Source of Truth?

Instead of maintaining a `GateSupportSet` that duplicates what traits already express:

- **Compile-time safety**: `CliffordGateable` bound ensures Clifford methods exist
- **No redundancy**: Traits already define supported gates
- **Clear semantics**: `rotations()` constructor requires `ArbitraryRotationGateable`

### Why Single `run()` Method?

Previous design had `run()` and `run_with_rotations()`. Now:

- Constructor determines capabilities (`new()` vs `rotations()`)
- Same `run()` method for all cases
- Runtime behavior matches compile-time constraints

### Why Function Pointers for Overrides?

`GateOverrides` uses `fn` pointers instead of `Box<dyn Fn>`:

- Zero allocation overhead
- Simpler lifetime management
- Sufficient for most use cases

For closures that capture state, wrap in a function that accesses shared state.
