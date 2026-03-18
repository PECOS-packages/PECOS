# CircuitRunner

`CircuitRunner` is for when you want to run circuits directly -- either because
`sim_neo` is more than you need, or because you want step-by-step control over
execution.

## Simple Circuit

```rust
use pecos_neo::prelude::*;
use pecos_qsim::SparseStab;

let circuit = CommandBuilder::new()
    .pz(0).pz(1)
    .h(0).cx(0, 1)
    .mz(0).mz(1)
    .build();

let mut state = SparseStab::new(2);
let mut runner = CircuitRunner::<SparseStab>::new();
let outcomes = runner.apply_circuit(&mut state, &circuit)?;
```

## With Noise

```rust
let mut runner = CircuitRunner::<SparseStab>::new()
    .with_noise(noise)
    .with_seed(42);
let outcomes = runner.apply_circuit(&mut state, &circuit)?;
```

## Custom Gates (GateId-based)

Use `OpBuilder` and `GateDefinitions` for custom or user-defined gates:

```rust
let definitions = GateDefinitions::new();
let circuit = OpBuilder::new()
    .pz(QubitId(0))
    .h(QubitId(0))
    .cx(QubitId(0), QubitId(1))
    .mz(QubitId(0), ResultId(0))
    .build();

let mut runner = CircuitRunner::<SparseStab>::with_definitions(definitions);
let outcomes = runner.run(&mut state, &circuit)?;
```

## Rotation Gates

Use the `rotations()` constructor for T gates, RX, RY, RZ, etc.:

```rust
use pecos_qsim::StateVec;

let circuit = OpBuilder::new()
    .pz(QubitId(0))
    .t(QubitId(0))
    .rz(QubitId(0), Angle64::QUARTER_TURN)
    .mz(QubitId(0), ResultId(0))
    .build();

let mut state = StateVec::new(1);
let mut runner = CircuitRunner::<StateVec>::rotations();
let outcomes = runner.run(&mut state, &circuit)?;
```

## Multiple Shots

Reset state and runner between shots:

```rust
let mut state = SparseStab::new(2);
let mut runner = CircuitRunner::<SparseStab>::new().with_seed(42);

for _ in 0..1000 {
    state.reset();
    runner.reset();
    let outcomes = runner.apply_circuit(&mut state, &circuit)?;
    // process outcomes...
}
```

## Going Deeper

For gate overrides, decomposition, signal/event handlers, noise event details,
and full API reference, see the [developer CircuitRunner guide](../dev/runner.md).
