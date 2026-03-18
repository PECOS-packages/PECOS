# Design Patterns and Conventions

This document describes the design patterns, API conventions, and architectural decisions used throughout pecos-neo. Following these patterns ensures consistency and helps contributors understand the codebase.

## API Hierarchy: Choosing the Right Entry Point

pecos-neo provides multiple APIs at different abstraction levels. Choose based on your needs:

### Decision Tree

```
Need to run a quantum circuit simulation?
│
├─► Simple case (standard gates, basic noise)?
│   └─► Use sim_neo() - highest level, batteries included
│
├─► Need non-Clifford gates (T, rotations)?
│   └─► Use sim_neo() with .quantum(state_vector()) -- rotations auto-enabled
│
├─► Need custom gates or decompositions?
│   └─► Use sim_neo() with .gate_definitions()
│
├─► Need gate overrides (swap implementations at runtime)?
│   └─► Use sim_neo() with .gate_overrides() (built-in backends)
│       or CircuitRunner with GateOverrides (custom backends, closures)
│
├─► Need fine-grained control over execution?
│   └─► Use CircuitRunner (GateType-based) or ProgramRunner
│
├─► Estimating rare event probabilities?
│   │
│   ├─► P ~ 10^-3 to 10^-6?
│   │   └─► Use sim_neo() with importance_sampling() orchestrator
│   │
│   └─► P ~ 10^-6 or smaller?
│       └─► Use SubsetSimulation or ProperSubsetSimulation
│
└─► Need population-based simulation (splitting, cloning)?
    └─► Use World<S> with ECS components
```

### API Comparison

| API | Abstraction | Use Case |
|-----|-------------|----------|
| `sim_neo()` | Highest | Standard simulations, custom gates, gate overrides, event handlers, quick prototyping |
| `CircuitRunner` | High | Direct simulator access, closure-based overrides, custom execution logic |
| `ProgramRunner` | Medium | Programs with classical control flow |
| `ImportanceSamplingRunner` | Medium | Direct importance sampling control |
| `World<S>` | Low | Population simulation, trajectory management |

### sim_neo() - The Recommended Entry Point

```rust
use pecos_neo::tool::{sim_neo, importance_sampling};

// Simple case
let results = sim_neo(circuit)
    .shots(1000)
    .seed(42)
    .run();

// With noise
let results = sim_neo(circuit)
    .depolarizing(0.01)
    .shots(1000)
    .run();

// With importance sampling
let results = sim_neo(circuit)
    .orchestrator(importance_sampling()
        .with_p1(0.001)
        .with_boost(10.0))
    .shots(10000)
    .run();

// Parallel execution
let results = sim_neo(circuit)
    .workers(4)
    .shots(1000)
    .run();

// Parallel execution with noise
let results = sim_neo(circuit)
    .depolarizing(0.01)
    .workers(4)
    .shots(1000)
    .run();

// With custom gate definitions
let defs = GateDefinitions::new();
let results = sim_neo(circuit)
    .gate_definitions(defs)
    .shots(1000)
    .run();

// State vector with non-Clifford gates (rotations auto-enabled)
let results = sim_neo(circuit)
    .quantum(state_vector())
    .shots(1000)
    .run();

// Control decomposition depth for deeply nested custom gates
let results = sim_neo(circuit)
    .max_decomp_depth(20)
    .shots(1000)
    .run();

// Gate overrides (swap gate implementations at runtime)
let overrides = GateOverrides::<SparseStab>::new()
    .register(gates::X, |sim, _angles, qubits| {
        // Custom implementation
        true
    });
let results = sim_neo(circuit)
    .gate_overrides(overrides)
    .shots(1000)
    .run();
```

### Event Handlers via sim_neo()

Event handlers (gate and signal handlers) can be passed through `sim_neo()` using
`EventHandlers`, which works with both sequential and parallel execution:

```rust
use pecos_neo::prelude::*;
use pecos_neo::tool::sim_neo;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

let gate_count = Arc::new(AtomicUsize::new(0));
let c = gate_count.clone();

let handlers = EventHandlers::new()
    .on_before_gate(move |_ctx| {
        c.fetch_add(1, Ordering::Relaxed);
        NoiseResponse::None
    });

let results = sim_neo(circuit)
    .event_handlers(handlers)
    .workers(4)  // handlers are cloned per worker
    .shots(1000)
    .run();
```

### When to Use CircuitRunner

Use `CircuitRunner` directly when you need:
- Direct simulator access between shots
- Custom execution logic beyond what `sim_neo()` provides

```rust
use pecos_neo::prelude::*;

let mut state = SparseStab::new(1);
let mut runner = CircuitRunner::<SparseStab>::new()
    .with_noise(noise)
    .with_seed(42);

let outcomes = runner.apply_circuit(&mut state, &commands)?;
```

### When to Use Lower-Level APIs

Use `CircuitRunner` or `ProgramRunner` when:
- You need direct simulator access between shots
- You're building custom execution logic
- You need gate overrides (`GateOverrides<S>`)
- You're integrating with existing infrastructure

```rust
// CircuitRunner for direct control
let mut state = SparseStab::new(n);
let mut runner = CircuitRunner::<SparseStab>::new()
    .with_noise(noise)
    .with_seed(42);

let outcomes = runner.apply_circuit(&mut state, &commands)?;
```

## Noise Model Selection

### Decision Tree

```
Need noise modeling?
│
├─► Simple depolarizing noise?
│   └─► Use .depolarizing(p) on sim_neo()
│
├─► Standard noise with different rates?
│   └─► Use GeneralNoiseModelBuilder
│       .with_p1(), .with_p2(), .with_p_meas()
│
├─► Need leakage, seepage, emission?
│   └─► Use CompositeNoiseModelBuilder
│       .with_p1_emission_ratio(), .with_leak_rate()
│
├─► Need gate-specific noise?
│   └─► Use GateDependentChannel or custom CompositeChannel
│
└─► Need custom noise logic?
    └─► Implement NoiseChannel trait or use composite primitives
```

### Noise Model Comparison

| Builder | Features | Complexity |
|---------|----------|------------|
| `.depolarizing(p)` | Uniform depolarizing | Simplest |
| `GeneralNoiseModelBuilder` | Separate p1/p2/p_meas | Simple |
| `CompositeNoiseModelBuilder` | Leakage, emission, seepage | Medium |
| Custom `CompositeChannel` | Gate-specific, conditional | Advanced |
| Flow primitives | Full decision trees | Most flexible |

## Builder Patterns

### The Builder-of-Builders Pattern

Complex configuration uses nested builders that compose naturally:

```rust
// Top-level builder accepts nested builders
sim_neo(circuit)
    .orchestrator(importance_sampling()  // Nested builder
        .with_p1(0.001)
        .with_boost(10.0))
    .quantum(sparse_stab())              // Another nested builder
    .noise(GeneralNoiseModelBuilder::new()  // And another
        .with_p1(0.001)
        .with_p2(0.01))
    .shots(1000)
    .run();
```

### When to Use Each Pattern

| Parameters | Pattern | Example |
|------------|---------|---------|
| 0-2 simple | Convenience method | `.shots(1000)`, `.seed(42)` |
| 0-2 with type | Free function → method | `.quantum(sparse_stab())` |
| 3+ related | Builder struct | `importance_sampling().with_p1().with_boost()` |
| Complex tree | Nested builders | Multiple levels of composition |

### Implementing New Builders

When adding a new configurable component:

```rust
// 1. Create builder struct with sensible defaults
#[derive(Debug, Clone)]
pub struct MyFeatureBuilder {
    param1: f64,
    param2: usize,
    param3: Option<String>,
}

impl MyFeatureBuilder {
    // 2. Constructor with defaults
    pub fn new() -> Self {
        Self {
            param1: 1.0,
            param2: 10,
            param3: None,
        }
    }

    // 3. Fluent setters (consume and return self)
    #[must_use]
    pub fn with_param1(mut self, value: f64) -> Self {
        self.param1 = value;
        self
    }

    // 4. Build method or Into implementation
    pub fn build(self) -> MyFeature {
        MyFeature { /* ... */ }
    }
}

// 5. Implement Default
impl Default for MyFeatureBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// 6. Implement From/Into for ergonomic use
impl From<MyFeatureBuilder> for MyFeature {
    fn from(builder: MyFeatureBuilder) -> Self {
        builder.build()
    }
}

// 7. Free function for discoverability
#[must_use]
pub fn my_feature() -> MyFeatureBuilder {
    MyFeatureBuilder::new()
}
```

### Accepting Builders in APIs

Use `impl Into<T>` for flexibility:

```rust
// Accepts both MyFeature and MyFeatureBuilder
pub fn configure(feature: impl Into<MyFeature>) -> Self {
    let feature = feature.into();
    // ...
}

// Usage:
.configure(my_feature().with_param1(2.0))  // Builder
.configure(existing_feature)                // Direct value
```

## The clone_box Pattern for Trait Objects

Trait objects (`Box<dyn Trait>`) cannot use `Clone` directly because `Clone`
requires `Sized`. The codebase uses a `clone_box()` method on each trait to
enable cloning trait objects:

```rust
pub trait NoiseChannel: Send + Sync {
    fn responds_to(&self, event: &NoiseEvent<'_>) -> bool;
    fn apply(&self, event: &NoiseEvent<'_>, ctx: &mut NoiseContext, rng: &mut PecosRng) -> NoiseResponse;
    fn name(&self) -> &'static str;

    /// Clone this channel into a boxed trait object.
    fn clone_box(&self) -> Box<dyn NoiseChannel>;
}
```

### Implementing clone_box for Custom Channels

Any custom `NoiseChannel` must implement `clone_box()`. The typical pattern is:

```rust
#[derive(Clone)]
struct MyChannel {
    probability: f64,
}

impl NoiseChannel for MyChannel {
    // ... responds_to, apply, name ...

    fn clone_box(&self) -> Box<dyn NoiseChannel> {
        Box::new(self.clone())
    }
}
```

This pattern is used by three traits:
- `NoiseChannel` -- noise channels in `ComposableNoiseModel`
- `EventHandler` -- state-tracking handlers (e.g., preparation/measurement tracking)
- `ContextObserver` -- observers that react to state changes (e.g., leakage)

### Type-Erased Cloning for Primitives

The `Primitive` trait (composite-based noise decision trees) also has `clone_box()`.
Composite primitives like `Prob<P>`, `When<C, T, E>`, and `Seq<P>` use
type-erased reconstruction in their `clone_box()` implementations. This avoids
requiring `P: Clone` on generic Primitive impls:

```rust
// Prob<P> does NOT require P: Clone for its Primitive impl.
// Instead, clone_box() reconstructs using Box<dyn Primitive>:
impl<P: Primitive> Primitive for Prob<P> {
    fn clone_box(&self) -> Box<dyn Primitive> {
        Box::new(Prob {
            probability: self.probability,
            inner: self.inner.clone_box(),  // Returns Box<dyn Primitive>
        })
    }
}
```

This works because `Box<dyn Primitive>` itself implements `Primitive`,
so the type-erased version is fully functional.

### Why clone_box Matters

`ComposableNoiseModel` implements `Clone` via `clone_box()` on its constituent
trait objects. This enables parallel Monte Carlo execution for noisy circuits --
each worker gets an independent clone of the noise model:

```rust
sim_neo(circuit)
    .depolarizing(0.01)
    .workers(4)     // Each worker clones the noise model
    .shots(10000)
    .run();
```

## Trait Bounds as Source of Truth

pecos-neo uses trait bounds to express capabilities at compile time rather than runtime checks.

### Philosophy

```rust
// GOOD: Trait bound expresses capability
impl<S: CliffordGateable> CircuitRunner<S> {
    pub fn new() -> Self { /* ... */ }
    pub fn with_definitions(defs: GateDefinitions) -> Self { /* ... */ }
}

// The rotations() constructor adds ArbitraryRotationGateable bound
impl<S: CliffordGateable + ArbitraryRotationGateable> CircuitRunner<S> {
    pub fn rotations() -> Self { /* ... */ }
}

// BAD: Runtime check for capability
impl CircuitRunner {
    pub fn run(&mut self) -> Result<_, Error> {
        if !self.supports_rotations() {  // Runtime check - avoid this
            return Err(Error::NoRotationSupport);
        }
    }
}
```

### Benefits

1. **Compile-time errors** - Misuse caught before runtime
2. **No redundant state** - Traits express what the simulator can do
3. **Clear API** - Constructor choice determines capabilities
4. **Zero overhead** - No runtime capability checks

### Applying This Pattern

When adding features that depend on simulator capabilities:

```rust
// Define capability as a trait bound
pub trait MyCapability {
    fn do_something(&mut self);
}

// Gate feature behind the bound
impl<S: CliffordGateable + MyCapability> CircuitRunner<S> {
    pub fn with_my_feature() -> Self { /* ... */ }
}
```

## Naming Conventions

### Types

| Category | Convention | Example |
|----------|------------|---------|
| Builders | `*Builder` | `SimNeoBuilder`, `ImportanceSamplingBuilder` |
| Configurations | `*Config` | `SubsetConfig`, `SimConfig` |
| Results | `*Result` or `*Results` | `SimulationResults` |
| Runners | `*CircuitRunner` | `CircuitRunner`, `CircuitRunner` |
| Channels | `*Channel` | `CompositeChannel`, `SingleQubitChannel` |

### Methods

| Category | Convention | Example |
|----------|------------|---------|
| Builder setters | `with_*` | `with_noise()`, `with_seed()` |
| Getters | field name or `get_*` | `outcomes()`, `get_bit()` |
| Conversions | `into_*`, `as_*`, `to_*` | `into_sim_neo_builder()` |
| Predicates | `is_*`, `has_*` | `is_empty()`, `has_weights()` |
| Actions | verb | `run()`, `reset()`, `build()` |

### Free Functions

Entry-point builders use lowercase snake_case functions:

```rust
pub fn sim_neo(input: impl SimNeoInput) -> SimNeoBuilder
pub fn importance_sampling() -> ImportanceSamplingBuilder
pub fn sparse_stab() -> SparseStabBuilder
pub fn state_vector() -> StateVecBuilder
```

## Error Handling

### Error Types

Use specific error enums for recoverable errors:

```rust
pub enum ExecutionError {
    NoDecomposition { gate_id: GateId },
    MaxDecompositionDepthExceeded,
}
```

### When to Panic vs Return Error

| Situation | Approach |
|-----------|----------|
| Programming error (bug) | `panic!` or `unreachable!` |
| Invalid user input | Return `Result` |
| Resource exhaustion | Return `Result` |
| Configuration error | Return `Result` or panic at build time |
| Invariant violation | `debug_assert!` in debug, assume valid in release |

### Error Messages

Include context that helps diagnose the issue:

```rust
// GOOD: Actionable context
panic!(
    "No program source set. Use sim_neo(circuit) or \
     sim_neo_builder().classical(builder) to provide a program."
);

// BAD: Vague
panic!("Invalid state");
```

## Testing Conventions

### Test Organization

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // Group related tests with comments
    // ========================================
    // Basic Functionality
    // ========================================

    #[test]
    fn test_feature_basic() { /* ... */ }

    #[test]
    fn test_feature_with_options() { /* ... */ }

    // ========================================
    // Edge Cases
    // ========================================

    #[test]
    fn test_feature_empty_input() { /* ... */ }
}
```

### Test Naming

```rust
fn test_<component>_<behavior>()
fn test_<component>_<scenario>_<expected>()

// Examples:
fn test_importance_sampling_basic()
fn test_importance_sampling_deterministic()
fn test_importance_sampling_produces_unbiased_estimates()
```

### Statistical Tests

For probabilistic behavior, use sufficient samples and reasonable tolerances:

```rust
#[test]
fn test_weighted_mean_approximates_true_value() {
    let num_shots = 2000;  // Enough for statistical significance

    // ... run simulation ...

    // Allow reasonable variance
    assert!(
        (estimated - expected).abs() < 0.1,
        "Expected ~{expected}, got {estimated}"
    );
}
```

## Module Organization

### Public API Surface

Expose types through `prelude` or module re-exports:

```rust
// In lib.rs or mod.rs
pub mod prelude {
    pub use crate::extensible::{
        GateDefinitions, GateId, GateSpec, OpBuilder,
    };
    // ... other commonly used types
}

// In tool/mod.rs - flat re-exports
pub use simulation::{
    sim_neo, importance_sampling, SimNeoBuilder,
    ImportanceSamplingBuilder, SimulationResults,
};
```

### Internal Organization

```
module/
├── mod.rs          # Public API, re-exports
├── builder.rs      # Builder types
├── runner.rs       # Execution logic
├── types.rs        # Core types and traits
└── tests.rs        # Or #[cfg(test)] mod tests in each file
```

## Documentation Standards

### Module-Level Docs

```rust
//! Brief description of the module.
//!
//! More detailed explanation of purpose and scope.
//!
//! # Examples
//!
//! ```rust
//! // Common usage pattern
//! ```
//!
//! # See Also
//!
//! - [`RelatedType`] - for X
//! - [`other_module`] - for Y
```

### Type-Level Docs

```rust
/// Brief one-line description.
///
/// Longer explanation if needed. Describe:
/// - What this type represents
/// - When to use it
/// - How it relates to other types
///
/// # Examples
///
/// ```rust
/// // Show typical usage
/// ```
pub struct MyType { /* ... */ }
```

### Method Docs

```rust
/// Brief description of what this method does.
///
/// # Arguments
///
/// * `param` - Description (only if not obvious)
///
/// # Returns
///
/// Description of return value (only if not obvious).
///
/// # Errors
///
/// Describe when this returns an error.
///
/// # Panics
///
/// Describe when this panics (if it can).
///
/// # Examples
///
/// ```rust
/// // Typical usage
/// ```
#[must_use]
pub fn my_method(&self) -> Result<T, E> { /* ... */ }
```

## Performance Considerations

### Prefer

- `Vec` over `HashMap` when keys are dense integers (use `GateId` indexing)
- `BTreeMap`/`BTreeSet` for deterministic iteration order
- `SmallVec` for typically-small collections
- `clone()` on `Arc` over deep cloning large structures
- Batch operations over individual operations

### Avoid

- Allocations in hot loops
- `HashMap` for small fixed sets of keys
- Deep trait object hierarchies
- Unnecessary `Box<dyn Trait>` when generics suffice

## Determinism

All simulation must be reproducible with the same seed:

```rust
// Use derived seeds for hierarchical determinism
let shot_seed = derive_seed(base_seed, &format!("shot_{idx}"));
let sim_seed = derive_seed(shot_seed, "simulator");
let noise_seed = derive_seed(shot_seed, "noise");

// Use BTreeMap/BTreeSet for ordered iteration
let entities: BTreeSet<EntityId> = /* ... */;

// Document any sources of non-determinism
```
