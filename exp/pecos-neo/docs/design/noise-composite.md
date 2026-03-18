# Composable Noise Flow Design

## Implementation Status

> **Status: Implemented** - The core design described in this document has been implemented. See [noise-usage-guide.md](../user-guides/noise.md) for practical usage.

### What's Implemented

| Component | Status | Location |
|-----------|--------|----------|
| Core primitives (Prob, When, Sample, Seq, SkipIf) | Complete | `composite/primitive.rs` |
| Conditions (Leaked, NotLeaked, OutcomeIs, etc.) | Complete | `composite/condition.rs` |
| Gate actions (Pauli, Leak, Seep, Inject, etc.) | Complete | `composite/action.rs` |
| Outcome actions (FlipOutcome, ForceOutcome) | Complete | `composite/action.rs` |
| Dynamic probability (ProbFn) | Complete | `composite/primitive.rs` |
| CompositeChannel integration | Complete | `composite/channel.rs` |
| CompositeNoiseModelBuilder | Complete | `composite/builder.rs` |
| Geometric sampling optimization | Complete | `composite/batch.rs` |
| Compiled primitives | Complete | `composite/compiled.rs` |
| Crosstalk channel | Complete | `composite/channel.rs` |
| Two-stage primitives | Complete | `composite/primitive.rs` |

### Additional Features (Beyond Original Design)

- **CompositeNoiseModelBuilder**: High-level builder API mirroring `GeneralNoiseModelBuilder`
- **Batch processing**: Geometric sampling for efficient low-probability channels
- **ForceOutcome**: Force measurement outcomes to specific values (not just flip)
- **Two-stage primitives**: Support for correlated multi-qubit noise
- **Partner tracking**: Context tracks "other qubit" in two-qubit gates

---

## Overview

This document describes a composable noise modeling system based on a small, fixed set of primitives that compose into decision trees. The design prioritizes:

- **Constrained**: Small primitive set, no abstraction explosion
- **Grounded**: Primitives map to real physical processes
- **Fast**: Tree structure enables early-exit and RNG batching
- **Flexible**: Primitives compose freely for custom noise models
- **Understandable**: Trees visualize naturally as flowcharts

## Design Philosophy

Most users want simple parameter-based noise (`depolarizing(0.01)`). Power users need deep customization. The design supports both through layers:

```
┌─────────────────────────────────────────┐
│  Presets (most users)                   │
│    IonTrap::standard(p1, p2, ...)       │
│    Superconducting::with_t1_t2(...)     │
├─────────────────────────────────────────┤
│  Primitives (power users)               │
│    prob, when, sample, seq, ...         │
├─────────────────────────────────────────┤
│  Custom actions (experts)               │
│    impl NoiseAction for MyNoise { }     │
└─────────────────────────────────────────┘
```

## Primitives

### Control Flow Primitives

| Primitive | Purpose | Example |
|-----------|---------|---------|
| `prob(p, action)` | With probability p, do action | `prob(0.01, pauli())` |
| `when(cond, then, else)` | Branch on state | `when(leaked, seep(), pauli())` |
| `sample([(w, a), ...])` | Weighted random choice | `sample([(0.75, pauli()), (0.25, emit())])` |
| `seq([a, b, c])` | Do all in order | `seq([check_leaked, apply_noise])` |
| `skip_if(cond)` | Early exit if condition | `skip_if(leaked)` |
| `per_qubit(action)` | Apply independently to each qubit | `per_qubit(fault_check)` |

### Conditions

| Condition | True when... |
|-----------|--------------|
| `leaked` | Qubit is in leaked state |
| `not_leaked` | Qubit is not leaked |
| `prepared` | Qubit has been prepared |
| `outcome_is(v)` | Measurement outcome equals v |

### Gate Actions

| Action | Effect |
|--------|--------|
| `pauli(weights)` | Inject random Pauli based on weights |
| `depolarize()` | Inject uniform random Pauli |
| `dephase()` | Inject Z error |
| `inject(gate)` | Inject specific gate |
| `remove_gate()` | Don't execute the triggering gate |
| `leak()` | Mark qubit as leaked |
| `seep(p)` | With prob p, unleak + random Pauli |
| `nothing()` | No-op (explicit do-nothing) |

### Outcome Actions

For measurement noise (operates on outcomes, not gates):

| Action | Effect |
|--------|--------|
| `flip_outcome()` | Flip measurement result 0↔1 |
| `force_outcome(v)` | Set outcome to specific value |
| `on_outcome(v, action)` | If outcome equals v, do action |

## Expressing Noise Models

### Preparation Noise

```
Prep Event:
1. Always clear leaked state (prep resets qubits)
2. Always execute prep gate
3. With prob p_prep:
   - With prob leak_ratio: leak
   - Otherwise: bit flip (X)
```

```rust
let prep_noise = seq([
    always(clear_leaked()),
    prob(p_prep,
        sample([
            (leak_ratio, leak()),
            (1.0 - leak_ratio, inject(X)),
        ])
    ),
]);
```

Flowchart:
```
Prep
│
├─▶ clear leaked state
│
└─▶ prob(p_prep)
    │
    └─▶ sample
        ├── leak_ratio ──▶ leak()
        └── 1-leak_ratio ──▶ X gate
```

### Single-Qubit Gate Noise

```
Single-Qubit Gate Event:
1. If qubit leaked: skip gate, but check for seepage
2. If not leaked: apply gate, then fault check
3. On fault:
   - If was leaked: seepage path
   - If not leaked: emission vs pauli branch
```

```rust
let sq_noise = seq([
    skip_if(leaked),  // Gate removed for leaked qubits
    prob(p1,
        when(leaked,
            // Leaked: seepage opportunity
            prob(p_seep, seep()),
            // Not leaked: emission vs pauli
            sample([
                (emission_ratio, emission_noise),
                (1.0 - emission_ratio, pauli(weights)),
            ])
        )
    ),
]);
```

Flowchart:
```
SingleQubitGate
│
├─▶ skip_if(leaked) ──▶ [gate removed]
│
└─▶ [gate executes]
    │
    └─▶ prob(p1)
        │
        └─▶ when(leaked)
            │
            ├─ YES ──▶ prob(p_seep) ──▶ seep()
            │
            └─ NO ──▶ sample
                      ├── emission_ratio ──▶ emission
                      └── 1-ratio ──▶ pauli
```

### Two-Qubit Gate Noise

Similar to single-qubit, with additions:
- Angle-dependent probability: `prob_fn(|gate| p2_angle_rate(gate.angle()))`
- Skip if ANY qubit leaked
- Two-qubit Pauli model
- Optional idle noise after

```rust
let tq_noise = seq([
    skip_if_any(leaked),
    prob_fn(|gate| p2_angle_error_rate(gate.angle()),
        when(any_leaked,
            per_qubit(prob(p_seep, seep())),
            sample([
                (emission_ratio, tq_emission_noise),
                (1.0 - emission_ratio, tq_pauli(weights)),
            ])
        )
    ),
    // Idle noise always applies (regardless of fault)
    prob(p2_idle, idle_pauli()),
]);
```

### Measurement Noise (Tricky Case #1)

**Why it's tricky:** Operates on outcomes, not gates. Outcome-dependent. Leaked qubits force outcome before flip noise applies.

```
Measurement Outcome Received:
1. If leaked: force outcome to 1
2. If outcome is 0: flip to 1 with prob p_meas_0
3. If outcome is 1: flip to 0 with prob p_meas_1
```

```rust
let meas_noise = seq([
    // Step 1: Leaked qubits forced to 1
    when(leaked, force_outcome(1)),

    // Step 2: Asymmetric flip noise (applies to possibly-forced outcome)
    on_outcome(0, prob(p_meas_0, flip_outcome())),
    on_outcome(1, prob(p_meas_1, flip_outcome())),
]);
```

Flowchart:
```
Measurement Outcome
│
├─▶ leaked? ── YES ──▶ force outcome = 1
│
├─▶ outcome = 0? ── YES ──▶ prob(p_meas_0) ──▶ flip to 1
│
└─▶ outcome = 1? ── YES ──▶ prob(p_meas_1) ──▶ flip to 0
```

**Key insight:** `seq` ensures ordering. Leaked check happens first, flip noise applies to the (possibly forced) outcome.

### Crosstalk (Tricky Case #2)

**Why it's tricky:** Non-local (affects OTHER qubits). Spans before/after boundary (inject measurements, wait, process outcomes). Coordinated operation.

**Solution:** Crosstalk is a special construct (not primitives), but its internal transitions use primitives.

```rust
let crosstalk = Crosstalk::builder()
    .trigger(Event::Measurement)

    // Who gets affected
    .victims(Victims::AllOtherPrepared)
    // Or: Victims::Nearby(radius)
    // Or: Victims::Custom(|ctx, measured| ...)

    // Probability of affecting each victim
    .global_probability(p_global)
    .local_probability(p_local)

    // What happens (uses primitives!)
    .outcome_transitions(|outcome| match outcome {
        0 => sample([
            (0.90, nothing()),
            (0.05, flip_state()),
            (0.05, leak()),
        ]),
        1 => sample([
            (0.85, nothing()),
            (0.10, flip_state()),
            (0.05, leak()),
        ]),
    })

    .build();
```

Flowchart:
```
Crosstalk (on Measurement)
│
├─▶ For each victim qubit:
│   │
│   └─▶ prob(p_crosstalk)
│       │
│       └─▶ [inject measurement, wait for outcome]
│           │
│           ├─▶ outcome = 0:
│           │   └─▶ sample: 90% nothing, 5% flip, 5% leak
│           │
│           └─▶ outcome = 1:
│               └─▶ sample: 85% nothing, 10% flip, 5% leak
```

**Why special construct?** Crosstalk is genuinely different from local noise. Forcing it into primitives would complicate them. Better: clean primitives + one honest special case.

### Idle Noise

Two independent components (both apply):

```rust
let idle_noise = seq([
    // Linear (T1-like): prob scales with duration
    per_qubit(
        skip_if(leaked),
        prob_fn(|gate| p_linear * gate.duration(),
            sample_pauli(t1_weights)
        )
    ),

    // Quadratic (T2-like): prob scales with duration^2
    per_qubit(
        skip_if(leaked),
        prob_fn(|gate| p_quad * gate.duration().powi(2),
            when(coherent_mode,
                inject(RZ(angle)),
                dephase()
            )
        )
    ),
]);
```

## Complete Model Assembly

```rust
let model = NoiseModel::new()
    // Local noise (primitives)
    .on_event(Event::Prep, prep_noise)
    .on_event(Event::SingleQubitGate, sq_noise)
    .on_event(Event::TwoQubitGate, tq_noise)
    .on_event(Event::MeasurementOutcome, meas_noise)
    .on_event(Event::Idle, idle_noise)

    // Non-local noise (special construct)
    .add_crosstalk(crosstalk)

    .build();
```

## Presets for Common Hardware

```rust
// Ion trap - most users just do this
let noise = IonTrapNoise::standard()
    .p1(0.001)
    .p2(0.01)
    .p_prep(0.001)
    .p_meas(0.01, 0.02)
    .emission_ratio(0.1)
    .build();

// Superconducting
let noise = SuperconductingNoise::standard()
    .t1(50e-6)
    .t2(30e-6)
    .gate_error(0.001)
    .readout_error(0.01, 0.02)
    .build();

// Custom - power users
let noise = NoiseModel::new()
    .on_event(Event::SingleQubitGate, my_custom_sq_noise)
    // ...
    .build();
```

## Performance

### Early Exit

The tree structure naturally supports early exit:
```rust
prob(0.01, ...)  // 99% of the time, exit here
```

### RNG Batching

For many gates, pre-roll fault checks:
```rust
// Processing 1000 gates with p = 0.01
// Generate all fault bits at once
let fault_mask = rng.batch_occurs(1000, 0.01);

// Only process ~10 faulted gates
for i in fault_mask.iter_ones() {
    apply_fault(&gates[i]);
}
```

### Compilation

Primitive trees can compile to optimized code:
- Eliminate virtual dispatch
- Inline small actions
- Optimal branch ordering

## Visualization

Multiple views for different needs:

```rust
// Parameter summary
noise.print_params();
// p1: 0.001, p2: 0.01, p_meas: (0.01, 0.02), ...

// Flowchart
noise.print_flowchart(Event::SingleQubitGate);
// ASCII tree shown above

// Mermaid (for docs)
println!("{}", noise.to_mermaid(Event::Measurement));

// Expected rates
noise.expected_rates();
// Single-qubit: 99.9% no error, 0.075% pauli, 0.025% emission

// Example trace
noise.trace_shot(seed: 42);
// Gate H(q0): no fault
// Gate CX(q0,q1): FAULT -> YI applied
// Meas(q0): outcome 0 -> flipped to 1
```

## Extensibility

Custom actions without modifying core:

```rust
// Custom action
struct AmplitudeDamping { gamma: f64 }

impl NoiseAction for AmplitudeDamping {
    fn apply(&self, qubit: QubitId, ctx: &mut Ctx, rng: &mut Rng) -> Response {
        // Custom logic
    }
}

// Custom condition
fn in_hot_zone(qubit: QubitId, ctx: &Ctx) -> bool {
    ctx.qubit_position(qubit).zone == Zone::Hot
}

// Use them
let noise = when(in_hot_zone,
    prob(0.1, AmplitudeDamping { gamma: 0.05 }),
    prob(0.01, depolarize()),
);
```

## Integration Considerations

This design relates to existing pecos-neo infrastructure:

| Existing | Relationship |
|----------|--------------|
| `ComposableNoiseModel` | Could use primitives internally |
| `NoiseChannel` trait | Primitives could implement this |
| `GeneralNoiseModelBuilder` | A preset that builds primitive trees |

Options:
1. **Internal implementation**: Primitives compile to existing channel interface
2. **Alternative API**: New way to build models, coexists with channels
3. **Replacement**: Migrate to primitives, deprecate channels

Recommendation: Start with option 1 (internal), validate the design works.

## Open Questions

1. **Validation**: Does expressing full GeneralNoiseModel in primitives work cleanly?
2. **Other hardware**: Have we missed patterns from neutral atoms, photonics, etc.?
3. **Macro DSL**: Would a macro make complex trees more readable?
4. **Serialization**: Define noise models in config files (YAML/JSON)?

## Summary

| Goal | Solution |
|------|----------|
| **Constrained** | Small fixed primitive set |
| **Grounded** | Primitives map to physical processes |
| **Fast** | Early exit, RNG batching, compilation |
| **Flexible** | Primitives compose freely |
| **Understandable** | Trees visualize as flowcharts |
| **Tricky cases** | Outcome primitives + Crosstalk special construct |

---

## Implementation Plan

> **Note**: Phases 1-3 are complete. Phase 4 (advanced optimization) is partially complete.

### Phase 1: Core Primitives + Validation - COMPLETE

**Goal:** Validate the design by implementing minimum viable primitives and expressing single-qubit gate noise.

**Deliverables:**
- Core primitive types: `Prob`, `When`, `Sample`, `Seq`, `SkipIf`
- Core conditions: `leaked`, `not_leaked`
- Core gate actions: `pauli`, `depolarize`, `leak`, `seep`, `inject`, `nothing`
- Basic `NoiseResponse` type
- Basic execution (no optimization)
- Test: Express single-qubit gate noise from GeneralNoiseModel
- Test: Verify behavior matches expected distribution

**Files created:**
```
crates/pecos-neo/src/noise/
  composite/
    mod.rs           # Module exports
    primitive.rs     # Primitive types (Prob, When, Sample, Seq, SkipIf)
    condition.rs     # Condition trait and built-in conditions
    action.rs        # GateAction trait and built-in actions
    response.rs      # CompositeResponse type
    channel.rs       # CompositeChannel integration
```

### Phase 2: Complete Coverage - COMPLETE

**Goal:** Express all of GeneralNoiseModel behavior.

**Deliverables:**
- Remaining gate actions: `seep` with probability, emission model sampling
- Outcome actions: `flip_outcome`, `force_outcome`, `on_outcome`
- Dynamic probability: `prob_fn(|gate| ...)`
- Crosstalk special construct: `CompositeCrosstalkChannel`
- All noise types: prep, two-qubit, measurement, idle
- Comparison tests against GeneralNoiseModel

**Validation criteria:**
- All GeneralNoiseModel behaviors expressible
- Statistical comparison tests pass
- Crosstalk works correctly

### Phase 3: Integration + Ergonomics - COMPLETE

**Goal:** Make it usable for real users.

**Deliverables:**
- Integration with `ComposableNoiseModel`: Complete via `CompositeChannel`
- `CompositeNoiseModelBuilder`: High-level builder API (implemented instead of hardware presets)
- Visualization: `describe_tree()` for primitive trees
- Documentation: See `docs/noise-usage-guide.md`

**Files created:**
- `composite/builder.rs` - `CompositeNoiseModelBuilder`
- `noise/introspection.rs` - Tree visualization

### Phase 4: Performance - PARTIAL

**Goal:** Make it fast for large-scale simulations.

**Deliverables:**
- Geometric sampling for batch processing: Complete (`composite/batch.rs`)
- Tree compilation: Complete (`composite/compiled.rs`)
- RNG batching: Complete (`GeometricSampler`)
- Benchmarks: Available in `benches/hot_path.rs`

**Remaining:**
- Further optimization based on profiling
- Hardware preset builders (IonTrap, Superconducting)

---

## Phase 1 Detailed Design

### File: `primitive.rs`

```rust
/// A noise primitive that can be composed into decision trees.
pub trait Primitive: Send + Sync {
    fn apply(
        &self,
        qubit: QubitId,
        ctx: &NoiseContext,
        rng: &mut PecosRng,
    ) -> NoiseResponse;
}

/// Probability gate: with probability p, execute inner primitive.
pub struct Prob<P: Primitive> {
    pub probability: f64,
    pub inner: P,
}

/// Conditional: if condition is true, execute then_branch, else else_branch.
pub struct When<C, T: Primitive, E: Primitive> {
    pub condition: C,
    pub then_branch: T,
    pub else_branch: E,
}

/// Weighted sample: choose one branch based on weights.
pub struct Sample<P: Primitive> {
    pub branches: Vec<(f64, P)>,
    // Normalized weights computed on construction
}

/// Sequential: execute all primitives in order, combine responses.
pub struct Seq<P: Primitive> {
    pub primitives: Vec<P>,
}

/// Early exit: if condition is true, return SkipGate response.
pub struct SkipIf<C> {
    pub condition: C,
}
```

### File: `condition.rs`

```rust
/// A condition that can be evaluated against noise context.
pub trait Condition: Send + Sync {
    fn evaluate(&self, qubit: QubitId, ctx: &NoiseContext) -> bool;
}

/// Built-in: qubit is leaked
pub struct Leaked;
impl Condition for Leaked {
    fn evaluate(&self, qubit: QubitId, ctx: &NoiseContext) -> bool {
        ctx.is_leaked(qubit)
    }
}

/// Built-in: qubit is not leaked
pub struct NotLeaked;
// ...
```

### File: `action.rs`

```rust
/// A terminal action that produces a noise response.
pub trait GateAction: Send + Sync {
    fn apply(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> NoiseResponse;
}

/// Apply random Pauli based on weights
pub struct Pauli {
    pub weights: PauliWeights,
}

/// Mark qubit as leaked
pub struct Leak;

/// No-op
pub struct Nothing;

// etc.
```

### File: `response.rs`

```rust
/// Result of applying noise primitive.
pub enum NoiseResponse {
    /// No noise applied
    None,
    /// Inject these gates after the current gate
    InjectGates(Vec<GateCommand>),
    /// Skip/remove the current gate
    SkipGate,
    /// Multiple responses to combine
    Multiple(Vec<NoiseResponse>),
}
```

### Test: Single-Qubit Noise

```rust
#[test]
fn test_sq_noise_expression() {
    // Express SQ noise from GeneralNoiseModel
    let sq_noise = Seq::new(vec![
        SkipIf::new(Leaked),
        Prob::new(p1,
            When::new(Leaked,
                Prob::new(p_seep, Seep),
                Sample::new(vec![
                    (emission_ratio, EmissionNoise::new(...)),
                    (1.0 - emission_ratio, Pauli::uniform()),
                ])
            )
        ),
    ]);

    // Verify statistical distribution
    let stats = test_distribution(&sq_noise, 100_000, seed);
    assert_approx_eq!(stats.no_fault_rate, 1.0 - p1, 0.01);
    assert_approx_eq!(stats.pauli_rate, p1 * (1.0 - emission_ratio), 0.01);
}
```
