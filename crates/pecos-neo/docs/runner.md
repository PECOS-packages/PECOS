# Runner: Unified Circuit Execution

## Overview

`Runner<S>` is the unified circuit runner for pecos-neo. It handles both `CommandQueue` (GateType-based) and `AdaptedSequence` (GateId-based) circuits, providing:

- **Trait-based native execution** - Compile-time checked via `CliffordGateable` / `ArbitraryRotationGateable`
- **Custom gate overrides** - Swap implementations for any gate, including core gates
- **Automatic decomposition** - Fall back to `GateDefinitions` when no native support
- **Composable noise** - Integrates with `ComposableNoiseModel` for event-driven noise
- **Signal dispatch** - Typed signals interleaved with gate execution
- **Gate event handlers** - Pluggable before/after hooks via `DispatchContext`

## Circuit Types

`Runner` supports two circuit representations through different methods:

| Circuit Type | Built With | Execution Method | Gate Identifier |
|---|---|---|---|
| `CommandQueue` | `CommandBuilder` | `execute()`, `run_shot()` | `GateType` (enum) |
| `AdaptedSequence` | `OpBuilder` | `run()`, `run_adapted_shot()` | `GateId` (u16) |

Use `CommandQueue` for simple circuits with core gates. Use `AdaptedSequence` when you need custom gates, decomposition, or gate overrides.

## Gate Execution Order

When `Runner` encounters a gate, it tries execution in this order:

```
1. Before-gate handlers + noise  (user handlers, then noise model)
2. Overrides     - Check GateOverrides registry
3. Clifford      - Try CliffordGateable trait methods
4. Rotation      - Try ArbitraryRotationGateable (if rotations() was used)
5. Decomposition - Expand via GateDefinitions
6. Error         - ExecutionError::NoDecomposition
7. After-gate handlers + noise   (noise model, then user handlers)
```

This is fail-fast: if a gate can't be handled, execution stops with an error.

## Basic Usage

### Simple Circuits (CommandQueue)

```rust
use pecos_neo::prelude::*;
use pecos_qsim::SparseStab;

let commands = CommandBuilder::new()
    .pz(0).pz(1)
    .h(0).cx(0, 1)
    .mz(0).mz(1)
    .build();

let mut runner = Runner::new(SparseStab::new(2));
let outcomes = runner.execute(&commands)?;
```

### Clifford Circuits with Custom Gates (AdaptedSequence)

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

let mut runner = Runner::with_definitions(SparseStab::new(2), definitions);
let outcomes = runner.run(&circuit)?;
```

### Circuits with Rotation Gates

```rust
use pecos_neo::prelude::*;
use pecos_qsim::StateVec;

let circuit = OpBuilder::new()
    .pz(QubitId(0))
    .rx(QubitId(0), Angle64::QUARTER_TURN)  // Rotation gate
    .t(QubitId(0))                           // T gate
    .mz(QubitId(0), ResultId(0))
    .build();

// Use rotations() constructor for native rotation support
let mut runner = Runner::rotations(StateVec::new(1));
let outcomes = runner.run(&circuit)?;  // Same run() method!
```

**Key insight**: The constructor (`new()` vs `rotations()`) determines which gates are native. The execution methods are the same.

## Gate Overrides

`GateOverrides` lets you provide custom implementations for any gate:

```rust
use pecos_neo::prelude::*;
use pecos_qsim::SparseStab;

let definitions = GateDefinitions::new();

// Register custom implementations
let overrides: GateOverrides<SparseStab> = GateOverrides::new()
    // Custom gate implemented as H
    .register(my_custom_gate, |sim, _angles, qubits| {
        sim.h(qubits);
        true
    })
    // Override core H gate (e.g., for debugging)
    .register(gates::H, |sim, _angles, qubits| {
        println!("H gate on {:?}", qubits);
        sim.h(qubits);
        true
    });

let mut runner = Runner::with_definitions(SparseStab::new(1), definitions)
    .with_overrides(overrides);
```

### Use Cases for Overrides

1. **Custom gates without decomposition** - Provide native implementation
2. **Debugging** - Add logging to specific gates
3. **Testing** - Replace gates with mocks
4. **Performance** - Optimized implementations for specific gates
5. **Temporary swaps** - Test alternative implementations

## Decomposition

When a gate has no override and isn't natively supported, `Runner` looks up its decomposition in `GateDefinitions`:

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

let mut runner = Runner::with_definitions(SparseStab::new(1), definitions);
let outcomes = runner.run(&circuit)?;  // my_gate is expanded automatically
```

### Decomposition Depth

To prevent infinite recursion, decomposition has a maximum depth (default: 10):

```rust
let runner = Runner::with_definitions(sim, definitions)
    .with_max_decomp_depth(20);  // Increase if needed
```

## Noise Integration

`Runner` integrates with `ComposableNoiseModel`:

```rust
use pecos_neo::prelude::*;

let noise = ComposableNoiseModel::new()
    .add_channel(SingleQubitChannel::depolarizing(0.001))
    .add_channel(TwoQubitChannel::depolarizing(0.01));

let mut runner = Runner::with_definitions(SparseStab::new(2), definitions)
    .with_noise(noise)
    .with_seed(42);

let outcomes = runner.run(&circuit)?;
```

### Noise Events

`Runner` emits `NoiseEvent::BeforeGate` and `NoiseEvent::AfterGate` with full gate metadata including `GateId`, enabling noise models to handle custom gates:

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
let mut runner = Runner::with_definitions(sim, definitions)
    .with_seed(42);

// run() keeps state for inspection
let outcomes = runner.run(&circuit)?;
println!("First: {:?}", outcomes);

// run_adapted_shot() returns outcomes and resets simulator + noise
for _ in 0..1000 {
    let outcomes = runner.run_adapted_shot(&circuit)?;
    // Process each shot...
}
```

For `CommandQueue` circuits:

```rust
let mut runner = Runner::new(SparseStab::new(2)).with_seed(42);

// run_shot() returns outcomes and resets noise (not simulator)
let outcomes = runner.run_shot(&commands)?;

// run_shot_fresh() also resets simulator -- optimized for Monte Carlo
for _ in 0..1000 {
    let outcomes = runner.run_shot_fresh(&commands)?;
    // Process each shot...
}
```

## API Reference

### Constructors

| Constructor | Trait Bound | Use Case |
|---|---|---|
| `Runner::new(sim)` | `CliffordGateable` | Simple circuits, default definitions |
| `Runner::with_definitions(sim, defs)` | `CliffordGateable` | Custom gates, decomposition |
| `Runner::rotations(sim)` | `+ ArbitraryRotationGateable` | Rotation gates, default definitions |
| `Runner::rotations_with_definitions(sim, defs)` | `+ ArbitraryRotationGateable` | Rotation gates + custom gates |

Native Clifford gates: H, X, Y, Z, SX, SY, SZ, CX, CY, CZ, SWAP, etc.
Additional rotation gates (with `rotations()`): T, Tdg, RX, RY, RZ, RXX, RYY, RZZ.

### Builder Methods

```rust
runner
    .with_noise(noise)           // Add noise model
    .with_seed(42)               // Set RNG seed
    .with_rng(rng)               // Set RNG directly
    .with_overrides(overrides)   // Add gate overrides
    .with_max_decomp_depth(20)   // Set decomposition limit
```

### Execution Methods

```rust
// CommandQueue (GateType-based)
runner.execute(&commands)?          // Execute, return &MeasurementOutcomes
runner.run_shot(&commands)?         // Execute, return owned outcomes, reset noise
runner.run_shot_fresh(&commands)?   // Reset simulator + run_shot (for Monte Carlo)

// AdaptedSequence (GateId-based)
runner.run(&circuit)?              // Execute, return &MeasurementOutcomes
runner.run_adapted_shot(&circuit)? // Execute, return owned outcomes, reset sim + noise
```

### Signal and Event Handlers

Handlers can be registered directly on `Runner`, or built via `EventHandlers`
for use with `sim_neo()` (including parallel workers):

```rust
// --- Via EventHandlers (works with sim_neo, cloneable) ---
let handlers = EventHandlers::new()
    .on_before_gate(|ctx| NoiseResponse::None)
    .on_signal(|sig: &MySignal| { /* observe */ });

// Pass to sim_neo (cloned per worker in parallel mode)
sim_neo(circuit).event_handlers(handlers).workers(4).shots(1000).run();

// Or merge into a Runner
let runner = Runner::new(sim).with_event_handlers(handlers);

// --- Direct registration on Runner ---
runner.on_signal::<MySignal>(|sig| { /* observe */ });
runner.on_signal_with_response::<MySignal>(|sig, ctx| NoiseResponse::None);

// Gate event handlers (return NoiseResponse)
runner.on_before_gate(|ctx| NoiseResponse::None);
runner.on_after_gate(|ctx| NoiseResponse::None);
runner.on_before_measurement(|ctx| NoiseResponse::None);
runner.on_after_measurement(|ctx| NoiseResponse::None);
runner.on_after_preparation(|ctx| NoiseResponse::None);
runner.on_idle(|ctx| NoiseResponse::None);

// Priority variants available for all gate event handlers:
runner.on_before_gate_with_priority(10, |ctx| NoiseResponse::None);
```

### Inspection

```rust
runner.simulator()              // &S - read-only simulator access
runner.simulator_mut()          // &mut S - mutable simulator access
runner.definitions()            // &GateDefinitions
runner.has_rotation_support()   // bool - was rotations() used?
runner.has_override(gate_id)    // bool - is gate overridden?
```

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
