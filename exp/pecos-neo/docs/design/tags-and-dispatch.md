# Signals and Pluggable Dispatch Design

## Implementation Status

> **Status: Implemented** - Phases 1-5 are complete. Signal infrastructure,
> signal dispatch in the runner, and gate event handlers with `DispatchContext`
> are all implemented and tested. See implementation notes at each phase below.

---

## Motivation

The current simulation pipeline has a rigid flow:

```
ClassicalEngine -> CommandQueue (gates) -> CircuitRunner -> NoiseModel + Simulator
```

This works well for standard quantum simulation, but real-world use cases need
richer communication between pipeline participants:

- A classical engine annotating qubits with zone temperatures or timing metadata
- An orchestrator marking QEC round boundaries for importance sampling sync points
- A noise model receiving backend-specific hints (e.g., "use approximate mode")
- A simulator receiving custom instructions that aren't standard gates
- Per-qubit calibration data flowing from a device model

These are not gates. They are **metadata that flows alongside gates** and must be
interpreted by specific consumers. Today, the only way to add a new consumer to the
pipeline is to modify `CircuitRunner`'s inner loop, which is fragile and doesn't compose.

### Design Goals

- **Simple**: One concept ("signals"), one API, minimal cognitive load.
- **Fast**: Zero overhead when no signals are present. Efficient dispatch when they are.
- **Scalable**: Works with millions of shots and large command queues.
- **Ordered**: Signals maintain position relative to gates in the command stream.
- **Flexible**: Simple scalar signals and complex structured signals use the same API.
- **Composable**: New consumers can be added without modifying the runner.
- **Bevy-inspired**: Follows Bevy 0.17's communication patterns where they fit,
  adapted for the constraints of high-performance quantum simulation.

---

## Background: Bevy 0.17 and 0.18

Bevy 0.17 separated two communication patterns that were previously conflated:

| Mechanism | Bevy Name | Characteristics |
|-----------|-----------|-----------------|
| Buffered, ordered, batch-processed | **Message** (`MessageWriter`/`MessageReader`) | Multiple independent readers, deterministic ordering, high throughput |
| Reactive, immediate, entity-scoped | **Event + Observer** (`Trigger`/`On<T>`) | Fire-and-forget, no ordering guarantees, one-to-many |
| Component lifecycle, always-on | **Hook** (`on_add`/`on_remove`) | Innate behavior, can't be removed |

Crucially, Bevy uses **typed structs** for all communication -- the Rust type IS
the routing key. There is no generic key-value container. Each message/event type
is a concrete struct with whatever fields it needs:

```rust
// Bevy style: the type is the key, fields are the data
#[derive(Event)]
struct ZoneTemperature(f64);

#[derive(Event)]
struct CalibrationData { rates: [f64; 8], num_qubits: u8 }
```

This sidesteps the "what data shapes should we support" question entirely --
users define whatever struct they need.

### Mapping to pecos-neo

| Bevy | pecos-neo |
|------|-----------|
| `Message<T>` (buffered, ordered) | Signals in the command stream |
| `Event + Observer` (reactive) | `NoiseChannel` trait (already exists) |
| `Hook` (lifecycle) | Gate registration (already exists) |

The signal system maps to Bevy's **Messages**: typed data produced by one
participant, consumed by one or more others in deterministic order.

The existing `NoiseChannel` trait already implements the **Observer** pattern:
channels self-select which events they handle via `responds_to()`, and the runner
broadcasts events without knowing which channels will respond.

---

## Design Overview

Two changes to the system:

1. **Signals**: Typed, user-defined data that flows alongside gates in the command
   stream. Simple newtypes for common cases, arbitrary structs for complex cases --
   all through the same API.

2. **Pluggable dispatch points**: The runner's inner loop becomes extensible so
   new consumers (signal handlers, telemetry, importance sampling hooks) can register
   without modifying `CircuitRunner`.

```
                    ┌────────────────────────────────┐
                    │        CommandQueue             │
                    │  ┌──────────────────────────┐   │
                    │  │ commands: Vec<GateCmd>    │   │  ← hot path, contiguous
                    │  └──────────────────────────┘   │
                    │  ┌──────────────────────────┐   │
                    │  │ signals: SignalStore      │   │  ← sparse, typed per-signal
                    │  └──────────────────────────┘   │
                    └───────────────┬─────────────────┘
                                    │
                    ┌───────────────▼─────────────────┐
                    │         CircuitRunner              │
                    │                                 │
                    │  for each item (ordered):       │
                    │    Gate   → dispatch points:    │
                    │      ├─ before_gate handlers    │
                    │      ├─ simulator.execute()     │
                    │      └─ after_gate handlers     │
                    │    Signal → dispatch points:    │
                    │      ├─ noise signal handlers   │
                    │      ├─ simulator hooks         │
                    │      └─ other handlers          │
                    └─────────────────────────────────┘
                                    │
                    Handlers registered at setup:
                    ├─ NoiseChannel (existing)
                    ├─ Signal handlers (new, typed)
                    ├─ Simulator hooks (new, optional)
                    └─ Telemetry / instrumentation (new)
```

---

## Part 1: Signals

### Core Principle: One Concept, One API

The user sees one thing: **signals**. A signal is a typed piece of data that
flows in the command stream. The API is the same whether the signal is a simple
`f64` wrapper or a complex struct:

```rust
// Define signal types -- simple newtypes are the common case
#[derive(Signal, Copy, Clone)]
struct ZoneTemperature(pub f64);

#[derive(Signal, Copy, Clone)]
struct RoundBoundary(pub i64);

#[derive(Signal, Copy, Clone)]
struct ApproximateMode(pub bool);

// Complex structs use the same API when needed
#[derive(Signal, Clone)]
struct CalibrationData {
    pub rates: [f64; 8],
    pub num_qubits: u8,
}

// Send -- always the same
queue.signal(ZoneTemperature(300.0));
queue.signal(RoundBoundary(42));
queue.signal(CalibrationData { rates: [0.01; 8], num_qubits: 4 });

// Receive -- always the same
runner.on_signal::<ZoneTemperature>(|temp| {
    // temp.0 is f64, fully typed
});
```

No decision tree ("should I use a tag or a message?"), no generic enum to match
on, no cognitive overhead. Most signals are simple newtypes; complex structs are
available when needed but not the first thing you reach for.

### The Signal Trait

```rust
/// Marker trait for types that can be sent as signals in the command stream.
/// Derive with `#[derive(Signal)]` for automatic implementation.
///
/// # Guidelines
///
/// Most signals are simple newtypes wrapping a single value:
///
/// ```
/// #[derive(Signal, Copy, Clone)]
/// struct ZoneTemperature(pub f64);
/// ```
///
/// For more complex data, use a struct with named fields:
///
/// ```
/// #[derive(Signal, Clone)]
/// struct CalibrationData {
///     pub rates: [f64; 8],
///     pub num_qubits: u8,
/// }
/// ```
pub trait Signal: Any + Send + Sync + Clone + 'static {
    /// Human-readable name for debugging and registration.
    fn name() -> &'static str;
}
```

The trait requires `Clone` (not `Copy`) so that both simple and complex signals
work through the same API. Simple newtypes that derive `Copy` get `Clone` for
free. The `Any` bound enables type-erased storage with typed retrieval.

### Storage: Typed Per-Signal, Sparse Overlay

Signals are stored in a type-erased heterogeneous container alongside the gate
command array. Each signal type gets its own `Vec<(u32, S)>`, stored contiguously
(DOD-friendly). This follows the SoA pattern already used in the ECS `World`.

```rust
pub struct CommandQueue {
    commands: Vec<GateCommand>,

    /// Typed signal storage. Each signal type gets a separate Vec<(u32, S)>
    /// keyed by TypeId. Empty when no signals are present (common case).
    signals: SignalStore,
}

/// Type-erased storage for heterogeneous signal types.
/// Each registered signal type gets its own contiguous Vec.
struct SignalStore {
    /// TypeId -> Box<dyn SignalVec> where SignalVec wraps Vec<(u32, S)>
    channels: HashMap<TypeId, Box<dyn SignalVec>>,
    /// Total signal count across all types (for fast is_empty check).
    total_count: usize,
}
```

Internally, each channel is a typed vector behind a trait object:

```rust
trait SignalVec: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn len(&self) -> usize;
    fn positions(&self) -> &[u32];   // for interleaved iteration
}

struct TypedSignalVec<S: Signal> {
    entries: Vec<(u32, S)>,
}

impl<S: Signal> SignalVec for TypedSignalVec<S> { ... }
```

#### Performance

- **No signals (common case)**: `signals.total_count == 0`, runner uses
  `iter_gates()` directly. Zero overhead.
- **Few signal types**: `HashMap<TypeId, _>` lookup is O(1). Each type's
  entries are contiguous.
- **Interleaved iteration**: Merge-sort across gate positions and signal
  positions. Cost proportional to number of signals, not number of gates.

### CommandQueue API

```rust
impl CommandQueue {
    /// Push a signal at the current position (after the last pushed command).
    pub fn signal<S: Signal>(&mut self, signal: S) { ... }

    /// Push a signal at a specific command index.
    pub fn signal_at<S: Signal>(&mut self, index: u32, signal: S) { ... }

    /// Iterate only gate commands (ignoring signals).
    pub fn iter_gates(&self) -> impl Iterator<Item = &GateCommand> { ... }

    /// Check if any signals are present.
    pub fn has_signals(&self) -> bool { self.signals.total_count > 0 }

    /// Iterate signals of a specific type.
    pub fn iter_signals<S: Signal>(&self) -> impl Iterator<Item = (u32, &S)> { ... }
}
```

### CommandBuilder Integration

```rust
let queue = CommandBuilder::new()
    .pz(0).pz(1).pz(2).pz(3)
    .signal(ZoneTemperature(300.0))      // zone A
    .h(0).h(1)
    .cx(0, 1)
    .signal(ZoneTemperature(350.0))      // zone B
    .h(2).h(3)
    .cx(2, 3)
    .mz(0).mz(1).mz(2).mz(3)
    .build();
```

### NoiseEvent Integration

A new variant on the existing enum enables noise channels to respond to signals.
Since `NoiseEvent` is a closed enum and signals are open-typed, the variant carries
type-erased data with typed access:

```rust
pub enum NoiseEvent<'a> {
    // ... existing variants unchanged ...

    /// A signal from the command stream.
    /// Use `signal::<T>()` for typed access.
    Signal {
        type_id: TypeId,
        data: &'a dyn Any,
    },
}

impl<'a> NoiseEvent<'a> {
    /// Try to extract a signal of a specific type.
    /// Returns `None` if this event is not a Signal or is a different type.
    pub fn signal<S: Signal>(&self) -> Option<&S> {
        match self {
            NoiseEvent::Signal { data, .. } => data.downcast_ref::<S>(),
            _ => None,
        }
    }

    /// Check if this is a signal of a specific type.
    pub fn is_signal<S: Signal>(&self) -> bool {
        matches!(self, NoiseEvent::Signal { type_id, .. } if *type_id == TypeId::of::<S>())
    }
}
```

Noise channels use typed access:

```rust
impl NoiseChannel for TemperatureAwareNoise {
    fn responds_to(&self, event: &NoiseEvent<'_>) -> bool {
        event.is_signal::<ZoneTemperature>() || matches!(event, NoiseEvent::AfterGate { .. })
    }

    fn apply(&self, event: &NoiseEvent<'_>, ctx: &mut NoiseContext, rng: &mut PecosRng) -> NoiseResponse {
        if let Some(temp) = event.signal::<ZoneTemperature>() {
            self.current_temp.set(temp.0);
            return NoiseResponse::None;
        }

        if let NoiseEvent::AfterGate { .. } = event {
            let rate = self.base_rate * (self.current_temp.get() / 300.0);
            // ... apply noise at adjusted rate ...
        }

        NoiseResponse::None
    }

    fn name(&self) -> &'static str { "temperature_aware" }
}
```

#### Dispatch Filtering

For O(1) filtering, noise channels can register which signal `TypeId`s they
respond to, matching the existing `BitVec`-based filtering for `GateId`. The
`responds_to()` method checks a `HashSet<TypeId>` or similar.

---

## Part 2: Pluggable Dispatch Points

### Problem

The current `CircuitRunner` has a hardcoded inner loop:

```rust
// Current: tightly coupled
for command in queue.iter() {
    let before = self.noise.emit(NoiseEvent::before_gate(...), rng);
    // handle skip, leakage, etc.
    self.simulator.execute(gate);
    let after = self.noise.emit(NoiseEvent::after_gate(...), rng);
    // handle injected errors, flips, etc.
}
```

Adding a new participant (signal consumer, telemetry, weight adjuster) requires
modifying this loop. This doesn't compose and couples the runner to every feature.

### Solution: Dispatch Points

The runner defines **dispatch points** -- named moments in the execution where
registered handlers are invoked. Handlers are registered at setup time and
called in priority order.

```
Gate execution timeline:

  ──┬──────────────┬───────────────┬──────────────┬─────────────
    │              │               │              │
    ▼              ▼               ▼              ▼
  BeforeGate    Execute        AfterGate     BetweenCmds
  handlers      (simulator)    handlers      (signal dispatch)
```

#### Handler Registration

> **Implementation note:** The actual API below supersedes the original proposal.
> Handlers receive `&DispatchContext` (read-only event info) and return
> `NoiseResponse` directly (no separate `DispatchResponse` type).

```rust
impl<S: CliffordGateable> CircuitRunner<S> {
    /// Register a typed observe-only signal handler.
    pub fn on_signal<Sig: Signal>(
        &mut self,
        handler: impl Fn(&Sig) + Send + Sync + 'static,
    ) -> &mut Self;

    /// Register a typed signal handler that returns NoiseResponse.
    pub fn on_signal_with_response<Sig: Signal>(
        &mut self,
        handler: impl Fn(&Sig, &DispatchContext<'_>) -> NoiseResponse + Send + Sync + 'static,
    ) -> &mut Self;

    /// Register a gate event handler (default priority 0).
    pub fn on_before_gate(
        &mut self,
        handler: impl Fn(&DispatchContext<'_>) -> NoiseResponse + Send + Sync + 'static,
    ) -> &mut Self;

    /// Register a gate event handler with explicit priority.
    pub fn on_before_gate_with_priority(
        &mut self,
        priority: i32,
        handler: impl Fn(&DispatchContext<'_>) -> NoiseResponse + Send + Sync + 'static,
    ) -> &mut Self;

    // Same pattern for: on_after_gate, on_before_measurement,
    // on_after_measurement, on_after_preparation, on_idle
}
```

#### Dispatch Context

Handlers receive read-only context with event data and optional noise state:

```rust
/// Context provided to gate event dispatch handlers.
pub struct DispatchContext<'a> {
    pub gate_type: GateType,
    pub qubits: &'a [QubitId],
    pub angles: &'a [Angle64],
    pub gate_id: Option<GateId>,
    pub outcomes: Option<&'a [bool]>,
    pub duration: Option<TimeUnits>,
    pub noise_context: Option<&'a NoiseContext>,
}
```

Fields not relevant to the current event are `None`/empty. Since handlers are
registered per-event-type, they know which fields are populated.

Handlers return `NoiseResponse` directly -- no separate `DispatchResponse`
type is needed since the response semantics are identical.

#### Backward Compatibility

The existing API continues to work unchanged:

```rust
// This still works -- noise model runs alongside gate event handlers
let runner = CircuitRunner::<SparseStab>::new()
    .with_noise(noise_model);
```

The `ComposableNoiseModel` stays intact as a dedicated subsystem. Gate event
handlers complement it -- user before-handlers run first, then noise model,
then user after-handlers. Responses from both are combined. When no gate
handlers are registered, the dispatch goes directly to the noise model with
zero overhead.

### CircuitRunner Inner Loop (Revised)

```rust
// Pseudocode -- the actual implementation manages interleaved iteration
for item in queue.iter_interleaved() {
    match item {
        CommandItem::Gate(cmd) => {
            // Dispatch: before_gate
            let response = self.dispatch_before_gate(cmd);
            if response.should_skip() { continue; }

            // Execute on simulator
            self.execute_gate(cmd);

            // Dispatch: after_gate
            let response = self.dispatch_after_gate(cmd);
            self.apply_response(response);
        }
        CommandItem::Signal(type_id, data) => {
            // Dispatch to typed signal handlers
            self.dispatch_signal(type_id, data);

            // Also emit as NoiseEvent::Signal for noise channels
            self.noise.emit(NoiseEvent::Signal { type_id, data }, rng);
        }
    }
}
```

The runner loop is now a generic dispatcher. What happens at each point is
determined by what's registered, not by hardcoded logic.

---

## Part 3: What We Are NOT Changing

### NoiseChannel trait stays as-is

The `NoiseChannel` trait (`responds_to`, `apply`, `try_apply`, `name`, `priority`)
is already the Observer pattern. It works well. Signals integrate through a new
`NoiseEvent::Signal` variant, not a new trait.

### No full schedule system

Bevy's schedule system enables flexible, user-defined execution ordering. In quantum
simulation, the execution order is physics-dictated: prepare, gate, noise, measure.
Making this configurable would add complexity without benefit.

### No archetype-based ECS storage

Bevy's archetype system optimizes for millions of entities with varied component
combinations. pecos-neo's entities are uniform (every entity has simulator + rng +
weight + noise context). `BTreeMap<EntityId, T>` per component is the right choice.

### GateCommand is unchanged

Signals are separate from gates. We do not add fields to `GateCommand` or create a
`Command` enum wrapping it. The gate array stays contiguous and cache-friendly.

---

## Usage Examples

### Simple: Zone Temperature

The most common pattern -- a newtype wrapping a single value:

```rust
#[derive(Signal, Copy, Clone)]
struct ZoneTemperature(pub f64);

// Send
let queue = CommandBuilder::new()
    .pz(0).pz(1)
    .signal(ZoneTemperature(300.0))
    .h(0).h(1)
    .cx(0, 1)
    .signal(ZoneTemperature(350.0))
    .h(0).h(1)
    .mz(0).mz(1)
    .build();

// Receive (noise channel)
impl NoiseChannel for TemperatureAwareNoise {
    fn responds_to(&self, event: &NoiseEvent<'_>) -> bool {
        event.is_signal::<ZoneTemperature>() || matches!(event, NoiseEvent::AfterGate { .. })
    }

    fn apply(&self, event: &NoiseEvent<'_>, ctx: &mut NoiseContext, rng: &mut PecosRng) -> NoiseResponse {
        if let Some(temp) = event.signal::<ZoneTemperature>() {
            self.current_temp.set(temp.0);
            return NoiseResponse::None;
        }
        if let NoiseEvent::AfterGate { .. } = event {
            let rate = self.base_rate * (self.current_temp.get() / 300.0);
            // ... apply noise at adjusted rate ...
        }
        NoiseResponse::None
    }

    fn name(&self) -> &'static str { "temperature_aware" }
}
```

### Simple: QEC Round Boundaries

```rust
#[derive(Signal, Copy, Clone)]
struct RoundBoundary(pub i64);

// In the circuit builder
for round in 0..num_rounds {
    builder.signal(RoundBoundary(round as i64));
    // ... syndrome extraction circuit ...
}

// Orchestrator reads round boundaries via signal handler
runner.on_signal::<RoundBoundary>(|round| {
    // round.0 is the round number
});
```

### Simple: Boolean Flags

```rust
#[derive(Signal, Copy, Clone)]
struct ApproximateMode(pub bool);

queue.signal(ApproximateMode(true));

runner.on_signal::<ApproximateMode>(|mode| {
    if mode.0 {
        // switch to approximate simulation
    }
});
```

### Advanced: Structured Calibration Data

For the less common case where complex data is needed:

```rust
#[derive(Signal, Clone)]
struct CalibrationData {
    pub qubit_rates: [f64; 8],
    pub coupling_strengths: [(u16, u16, f64); 4],
    pub num_qubits: u8,
}

queue.signal(CalibrationData {
    qubit_rates: [0.001, 0.002, 0.0015, 0.001, 0.003, 0.001, 0.002, 0.0012],
    coupling_strengths: [(0, 1, 0.01), (1, 2, 0.015), (2, 3, 0.01), (3, 0, 0.008)],
    num_qubits: 4,
});

runner.on_signal::<CalibrationData>(|cal| {
    for i in 0..cal.num_qubits as usize {
        // set per-qubit noise rates from cal.qubit_rates[i]
    }
});
```

### Multiple Consumers for the Same Signal

```rust
#[derive(Signal, Copy, Clone)]
struct RoundBoundary(pub i64);

// Noise model adjusts parameters per round
impl NoiseChannel for AdaptiveNoise {
    fn responds_to(&self, event: &NoiseEvent<'_>) -> bool {
        event.is_signal::<RoundBoundary>()
    }
    fn apply(&self, event: &NoiseEvent<'_>, ctx: &mut NoiseContext, rng: &mut PecosRng) -> NoiseResponse {
        if let Some(round) = event.signal::<RoundBoundary>() {
            // adjust noise parameters for this round
        }
        NoiseResponse::None
    }
    fn name(&self) -> &'static str { "adaptive" }
}

// Orchestrator also sees the same signal (observe-only)
runner.on_signal::<RoundBoundary>(|round| {
    // sync point for importance sampling
});
```

---

## Implementation Plan

### Phase 1: Signal Trait and `impl_signal!` Macro (pecos-core) -- DONE

Added the `Signal` trait to pecos-core with an `impl_signal!` macro
(`macro_rules!`, no proc-macro crate needed). No new dependencies,
no breaking changes.

**Files:**
- `crates/pecos-core/src/signal.rs` (new)
- `crates/pecos-core/src/lib.rs` (re-export `Signal`)
- `crates/pecos-core/src/prelude.rs` (re-export `Signal`)

### Phase 2: Signal Store and CommandQueue Integration (pecos-neo) -- DONE

Added `SignalStore` (type-erased heterogeneous container, SoA pattern) to
`CommandQueue`. Added `.signal()` to both `CommandQueue` and `CommandBuilder`.
Also added `has_signals()`, `iter_signals::<S>()`, `signal_at()`.

**Files:**
- `crates/pecos-neo/src/command.rs` (SignalStore field, signal methods)
- `crates/pecos-neo/src/command/builder.rs` (`.signal()` in fluent API)
- `crates/pecos-neo/src/command/signal_store.rs` (new: SignalStore, TypedSignalVec, SignalIter)

### Phase 3: NoiseEvent::Signal Variant (pecos-neo) -- DONE

Added `Signal { type_id: TypeId, data: &'a dyn Any }` variant to `NoiseEvent`.
Added typed access methods: `signal::<T>()`, `is_signal::<T>()`, `is_signal_event()`,
`from_signal()`. Updated the two exhaustive matches (`qubits()`, `apply_state_updates()`)
to handle the new variant. All existing noise channels are unaffected (they use
wildcard `_` arms).

**Files:**
- `crates/pecos-neo/src/noise.rs`

### Phase 4: Signal Dispatch in CircuitRunner (pecos-neo) -- DONE

Added `SignalHandlerRegistry` and `.on_signal::<T>()` to `CircuitRunner`.
Signals are dispatched during `execute()` using a cursor-based interleaving
approach (signals at position N fire between commands N-1 and N). Observe-only
handlers fire first, then signals are emitted to the noise model as
`NoiseEvent::Signal`. The external API (`with_noise`, `run_shot`, etc.)
stays unchanged.

**Files:**
- `crates/pecos-neo/src/runner.rs` (SignalHandlerRegistry, dispatch_signals_at)

### Phase 5: Gate Event Handlers and DispatchContext (pecos-neo) -- DONE

Added `DispatchContext` (gate info + optional read-only `NoiseContext`) and
`GateEventHandlers` (per-event-type `Vec<PrioritizedHandler>`) to `CircuitRunner`.
Users register closures via `on_before_gate`, `on_after_gate`,
`on_before_measurement`, `on_after_measurement`, `on_after_preparation`, and
`on_idle`. Handlers return `NoiseResponse` which is combined with noise model
responses. Also added `on_signal_with_response` for signal handlers that
return `NoiseResponse`.

Fast path: when no gate handlers are registered, dispatch goes directly to the
noise model with zero overhead.

**Files:**
- `crates/pecos-neo/src/runner.rs` (DispatchContext, GateEventHandlers, PrioritizedHandler)

### Phase 6: EventHandlers for sim_neo() -- DONE

Changed handler storage from `Box<dyn Fn>` to `Arc<dyn Fn>`, making
`GateEventHandlers` and `SignalHandlerRegistry` `Clone`. Added public
`EventHandlers` type wrapping both, with builder-pattern registration methods
mirroring CircuitRunner's `on_*` API. `CircuitRunner::with_event_handlers()` merges an
`EventHandlers` into the runner's registries.

Plumbed through `sim_neo()`: `SimNeoBuilder::event_handlers()` stores handlers
as a resource, applied at startup for both `SparseStab` and `StateVec`
backends. In parallel mode, handlers are cloned per worker via
`ParallelExecutionData`.

**Files:**
- `crates/pecos-neo/src/runner.rs` (Arc types, EventHandlers, with_event_handlers)
- `crates/pecos-neo/src/program.rs` (ProgramRunner::with_event_handlers)
- `crates/pecos-neo/src/tool/simulation.rs` (EventHandlersResource, builder method, parallel plumbing)

---

## Performance Considerations

### Hot Path (No Signals)

When `CommandQueue` has no signals (`signals.total_count == 0`), the runner
uses `iter_gates()` which iterates the `Vec<GateCommand>` directly. No
`HashMap` lookups, no type erasure, no overhead.

### Signal Dispatch

When signals are present:
- **Storage**: Each signal type is in its own contiguous `Vec<(u32, S)>`.
  No mixing of types, cache-friendly per-type iteration.
- **Type lookup**: `HashMap<TypeId, _>` is O(1) per type. Done once per
  type per queue iteration, not per signal.
- **Interleaved iteration**: Merge positions from gates and signals.
  Cost proportional to number of signals (typically few).
- **Handler dispatch**: Direct function call per registered handler.
  No virtual dispatch chain.

### Simple vs Complex Signals

Simple `Copy` signals (newtypes wrapping scalars) are cloned trivially --
a memcpy of a few bytes. Complex signals with arrays or non-Copy data
use `Clone`, which may allocate. This is the natural performance gradient:
simple data is fast, complex data costs more. No artificial distinction needed.

---

## Resolved Design Decisions

1. **Qubit context is per-signal-type, not trait-level.** The `Signal` trait does
   not require qubit targeting. Signal types that need qubit context include
   `QubitId`s as fields. Most signals (temperatures, round boundaries, flags)
   don't need them.

2. **Signals are per-entity; orchestrator gets its own store.** Each entity
   processes its own `CommandQueue` with its own signals -- this is the common
   case. The orchestrator can also hold signals in the `World` as a shared
   resource (e.g., a `SignalStore` component on a well-known entity or a
   dedicated `World` resource). This keeps orchestrator-level signals inside
   the `World` rather than as external state.

3. **Signals are one-directional.** The forward path is
   engine -> `CommandQueue` -> runner -> consumers. The existing backward
   paths already cover responses:
   - `NoiseResponse` (noise channels respond to events including signals)
   - `MeasurementOutcomes` (results flow back to the orchestrator)
   - Orchestrator inspection of per-shot results (weights, outcomes, etc.)

   If request/response semantics are needed, define two signal types and use
   the orchestrator as intermediary: `CalibrationRequest` flows forward in
   shot N, the orchestrator observes results, then injects `CalibrationUpdate`
   into shot N+1's `CommandQueue`. No special bidirectional infrastructure.

4. **Signals are transient per-shot.** Signals live in the `CommandQueue` and
   are consumed during shot execution. They do not persist across shots. The
   orchestrator can observe results and decide to inject new signals into
   subsequent shots, but the signals themselves are re-emitted each time.

5. **`Signal` trait and `impl_signal!` macro live in pecos-core.** The trait
   is minimal (`Any + Send + Sync + Clone + 'static`) and the macro is
   `macro_rules!`, so no proc-macro crate is needed. All signal infrastructure
   (storage, dispatch, `NoiseEvent` integration) lives in pecos-neo. If a
   proc-macro `#[derive(Signal)]` is wanted later, it can be added to a
   `pecos-derive` crate.

---

## References

- [Bevy 0.17 Release Notes](https://bevy.org/news/bevy-0-17/) -- Message/Event/Observer split
- [Bevy Observer Overhaul Design Doc](https://hackmd.io/@bevy/rk4S92hmlg) -- Observer architecture
- [Bevy 0.18 Release Notes](https://bevy.org/news/bevy-0-18/) -- Safe component access
- [Bevy Events - Unofficial Cheat Book](https://bevy-cheatbook.github.io/programming/events.html) -- Typed event patterns
- [noise-composite.md](noise-composite.md) -- Existing composable noise design
- [design-patterns.md](../dev/design-patterns.md) -- pecos-neo API conventions
