# Events, Signals, and Handlers

The simulation pipeline is event-driven. At each point in execution -- before
a gate, after a measurement, during idle time -- the runner emits an event.
[Noise channels](noise-channels.md) are the primary consumers of these events
(that's how noise gets applied), but you can also register your own handlers
to observe or modify execution.

## The Event Flow

During execution, the runner processes each event in this order:

1. **Your event handlers** run (via `EventHandlers`)
2. **Noise model event handlers** run (state tracking: prep, measurement, leakage)
3. **Noise channels** that respond to this event produce `NoiseResponse`s
4. **Observers** react to state changes from the responses

This means event handlers, noise channels, and signals all participate in the
same pipeline. A signal you inject into the command stream reaches both your
`on_signal` handlers and any noise channel that responds to it.

## Event Handlers

Register callbacks for gate and measurement events. They receive a
`DispatchContext` with the gate type, qubits, and (for measurements) outcomes:

```rust
use pecos_neo::prelude::*;

let handlers = EventHandlers::new()
    .on_before_gate(|ctx| {
        println!("About to run {:?} on {:?}", ctx.gate_type, ctx.qubits);
        NoiseResponse::None
    })
    .on_after_measurement(|ctx| {
        println!("Measured {:?}, outcomes: {:?}", ctx.qubits, ctx.outcomes);
        NoiseResponse::None
    });

sim_neo(circuit).event_handlers(handlers).shots(1000).run();
```

Handlers return `NoiseResponse` to modify execution -- not just observe it:

```rust
// Skip identity gates entirely
EventHandlers::new().on_before_gate(|ctx| {
    if ctx.gate_type == GateType::I {
        NoiseResponse::SkipGate
    } else {
        NoiseResponse::None
    }
});
```

### Available Hooks

| Method | When it fires |
|--------|---------------|
| `on_before_gate` | Before a gate executes |
| `on_after_gate` | After a gate executes |
| `on_before_measurement` | Before a measurement |
| `on_after_measurement` | After a measurement (outcomes available) |
| `on_after_preparation` | After state preparation |
| `on_idle` | During idle time |

All hooks have a `_with_priority` variant for controlling execution order
(higher priority runs first).

## Signals

Signals are typed data that flow alongside gates in the command stream. Define
a signal type, inject it into the circuit, and handle it during execution:

```rust
use pecos_core::{Signal, impl_signal};

#[derive(Copy, Clone, Debug)]
struct RoundBoundary(pub usize);
impl_signal!(RoundBoundary);

// Inject signals between gates
let circuit = CommandBuilder::new()
    .pz(0).h(0).cx(0, 1)
    .signal(RoundBoundary(1))
    .h(0).cx(0, 1)
    .signal(RoundBoundary(2))
    .mz(0).mz(1)
    .build();

// Handle them during execution
let handlers = EventHandlers::new()
    .on_signal::<RoundBoundary>(|round| {
        println!("Starting round {}", round.0);
    });
```

Noise channels can also react to signals -- see
[Signal-Reactive Channels](noise-channels.md#signal-reactive-channels) for
how to build channels that adjust their behavior based on custom triggers.

## Going Deeper

For the full `DispatchContext` struct, `on_signal_with_response`,
and the `NoisePlugin`/`EventHandler`/`ContextObserver` traits, see the
[signals and dispatch design doc](../design/tags-and-dispatch.md).
