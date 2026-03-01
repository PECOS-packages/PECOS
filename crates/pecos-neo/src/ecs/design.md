# ECS-Inspired Parallel Simulation Architecture

## Status: Implementation Complete (Phases 1-4)

This document captures our current thinking on the parallel simulation architecture
for pecos-neo. The goal is to apply DOD (Data-Oriented Design) and ECS (Entity
Component System) principles to create a flexible, performant system for:

- Standard Monte Carlo (millions of shots)
- Rare event simulation (splitting, subset simulation)
- Large-scale simulators (10K to 1M+ qubits)

## Problem Analysis

### Scale Requirements

| Dimension | Scale | Notes |
|-----------|-------|-------|
| Shots | Millions | Embarrassingly parallel |
| Qubits per shot | 10K - 1M+ | Mostly sequential execution |
| Parallelism | High | Need efficient multi-core utilization |

### Access Patterns

**Hot Path (per gate, billions of calls)**:
- Simulator state access/mutation
- Noise context queries (is qubit leaked? prepared?)
- RNG sampling for noise decisions
- Measurement outcome recording

**Cold Path (per shot or less frequent)**:
- Entity allocation/deallocation
- Seed derivation
- Weight updates (importance sampling)
- Splitting/pruning decisions
- Statistics aggregation

### Key Insight

The simulator internals (stabilizer tableau, state vector) are already optimized
for their access patterns. The ECS layer should manage the *population* of
simulators, not interfere with their hot paths.

```
┌─────────────────────────────────────────────────────────────┐
│                    ECS Layer (Cold)                          │
│  Entity management, seed derivation, coordination            │
├─────────────────────────────────────────────────────────────┤
│                    Execution Layer (Hot)                     │
│  ShotRunner + Simulator + NoiseContext                       │
│  (tight loop, no ECS overhead)                               │
└─────────────────────────────────────────────────────────────┘
```

## DOD Principles Applied

### 1. Separate Hot and Cold Data

```rust
// Hot: Co-located for cache efficiency during gate execution
struct ExecutionContext<S> {
    simulator: S,           // Quantum state
    noise_ctx: NoiseContext, // Leakage, prepared qubits
    rng: PecosRng,          // Noise sampling
}

// Cold: Managed by ECS World
struct EntityData {
    weight: SampleWeight,    // Importance sampling
    status: Status,          // Active/Complete/Pruned
    path: PathState,         // Branching program position
    outcomes: Outcomes,      // Accumulated measurements
}
```

### 2. Structure of Arrays (SoA) for Cold Data

The `ComponentStorage<T>` uses `BTreeMap<EntityId, T>` which gives:
- Deterministic iteration order
- Efficient per-component iteration
- Sparse storage (not all entities need all components)

For hot data, the simulator already uses optimized internal layouts.

### 3. Avoid Indirection in Hot Path

Current noise channels use `Box<dyn NoiseChannel>` with vtable dispatch.
For the hot path, consider:
- Enum dispatch instead of trait objects
- Monomorphization where possible
- Inline small channels

### 4. Batch Operations

Rather than processing one entity at a time, batch operations allow:
- Better cache utilization
- SIMD opportunities (future)
- Reduced scheduling overhead

## Parallelism Architecture

### Level 1: Worker Parallelism (Current Focus)

Each worker owns independent state. No shared memory between workers.

```
┌─────────────────────────────────────────────────────────────┐
│                      Coordinator                             │
│  - Distributes work to workers                               │
│  - Aggregates results                                        │
│  - Makes global decisions (splitting, termination)           │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
    ┌─────────┐          ┌─────────┐          ┌─────────┐
    │ Worker 0│          │ Worker 1│          │ Worker 2│
    │         │          │         │          │         │
    │ World   │          │ World   │          │ World   │
    │ + Shots │          │ + Shots │          │ + Shots │
    └─────────┘          └─────────┘          └─────────┘
```

**Pros**: Simple, no contention, scales well
**Cons**: Global decisions require synchronization

### Level 2: Coordinated Rare Event Simulation

For splitting/subset simulation, we need periodic coordination:

```
┌─────────────────────────────────────────────────────────────┐
│  Time →                                                      │
│                                                              │
│  ║══════════╗   ║══════════╗   ║══════════╗                 │
│  ║ Parallel ║   ║ Parallel ║   ║ Parallel ║                 │
│  ║ Execute  ║   ║ Execute  ║   ║ Execute  ║                 │
│  ║══════════╝   ║══════════╝   ║══════════╝                 │
│        │              │              │                       │
│        ▼              ▼              ▼                       │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐                 │
│  │  Sync    │   │  Sync    │   │  Sync    │                 │
│  │  Point   │   │  Point   │   │  Point   │                 │
│  └──────────┘   └──────────┘   └──────────┘                 │
│        │              │              │                       │
│  - Aggregate weights                                         │
│  - Evaluate splitting criteria                               │
│  - Redistribute entities                                     │
│  - Prune low-weight trajectories                            │
└─────────────────────────────────────────────────────────────┘
```

### Level 3: Within-Simulator Parallelism (Future)

For very large simulators (1M+ qubits), there may be opportunities for
parallel noise context updates. This is speculative and needs profiling.

## Proposed Components

### Core (Implemented)

```rust
// Entity identifier (lightweight, deterministic)
pub struct EntityId(pub u64);

// Component storage (BTreeMap for determinism)
pub struct ComponentStorage<T> { data: BTreeMap<EntityId, T> }

// World: entity + component management
pub struct World<S: CliffordGateable> {
    simulators: ComponentStorage<SimulatorComponent<S>>,
    rngs: ComponentStorage<RngComponent>,
    weights: ComponentStorage<WeightComponent>,
    // ...
    resources: Resources,  // Shared: seed, time
}

// Centralized seed management
pub struct SeedResource { base_seed: u64 }
```

### Parallel Coordinator (To Be Implemented)

```rust
/// Coordinates parallel execution across workers
pub struct ParallelCoordinator<S> {
    base_seed: u64,
    num_workers: usize,
    // ...
}

/// Configuration for parallel execution
pub struct ParallelConfig {
    num_workers: usize,
    shots_per_batch: usize,
    sync_interval: Option<usize>,  // For rare event simulation
}

/// Result from parallel execution
pub struct ParallelResult<T> {
    results: Vec<T>,
    stats: ExecutionStats,
}
```

### Rare Event Extensions (To Be Implemented)

```rust
/// Splitting criteria for rare event simulation
pub trait SplittingCriterion {
    fn should_split(&self, entity: EntityId, world: &World) -> Option<usize>;
}

/// Subset simulation level definition
pub struct SubsetLevel {
    threshold: f64,
    target_count: usize,
}
```

## Open Questions

### 1. Noise Context Location

Should `NoiseContext` be:
- **A)** Inside `ShotRunner` (current) - good for hot path co-location
- **B)** A separate component in World - more ECS-pure, but adds indirection
- **C)** Hybrid: hot bits in runner, cold metadata in World

**Current thinking**: Option A for standard simulation, maybe C for complex noise.

### 2. Simulator Cloning for Splitting

When splitting a trajectory:
- Clone the full simulator state (expensive for large states)
- Or checkpoint/restore mechanism
- Or lazy cloning with copy-on-write

**Current thinking**: Start with full clone, optimize if profiling shows need.

### 3. Work Distribution

For rare event simulation with varying entity counts:
- Static distribution (equal entities per worker)
- Dynamic work-stealing
- Centralized task queue

**Current thinking**: Start static, add work-stealing if load imbalance observed.

### 4. Memory Management

With millions of shots and large simulators:
- Pre-allocate entity pools?
- Reuse simulators across shots?
- Stream results to disk?

**Current thinking**: Measure memory pressure first, then optimize.

## Testing Strategy

### Correctness Tests

1. **Determinism**: Same seed produces identical results
   - Single-threaded baseline
   - Multi-threaded must match (with same seed distribution)

2. **Statistical validation**: Compare against known results
   - Repetition code logical error rates
   - Surface code threshold estimates

3. **Comparison with MonteCarloEngine**:
   - Same circuits, same noise, same seeds
   - Results should be statistically equivalent

### Performance Benchmarks

1. **Throughput**: Shots per second
   - Vary: num_qubits, num_workers, noise complexity
   - Compare: MonteCarloEngine vs new system

2. **Scaling**: Speedup vs worker count
   - Identify: saturation point, overhead sources

3. **Memory**: Peak usage and allocation patterns
   - Profile: large qubit counts, many shots

4. **Latency**: Time to first result
   - Important for interactive use cases

### Profiling Focus Areas

1. Gate execution hot path
2. Noise sampling overhead
3. Seed derivation cost
4. Entity management overhead
5. Synchronization costs (rare event simulation)

## Implementation Plan

### Phase 1: Foundation (Done)
- [x] EntityId, ComponentStorage
- [x] World with basic entity management
- [x] Centralized seed derivation
- [x] Entity cloning/splitting
- [x] Validation tests against MonteCarloEngine

### Phase 1.5: Validation (Done)

Comprehensive validation tests verify correctness:

**Engine Comparison Tests** (`tests/engine_comparison_test.rs`):
- MonteCarloRunner vs MonteCarloEngine statistical equivalence
- Determinism with `with_full_seed()` across runs
- Parallel execution produces correct distributions
- Bell state correlations, depolarizing noise, measurement errors

**Noise Comparison Tests** (`tests/noise_comparison_test.rs`):
- ComposableNoiseModel vs GeneralNoiseModel
- Single/two-qubit depolarizing, measurement, preparation errors
- Combined noise channels
- Idle noise with TimeScale

**Surface Code Comparison Tests** (`tests/surface_code_comparison_test.rs`):
- Repetition code (d=3) syndrome extraction
- Logical error rates vs rounds
- Temporal and spatial correlations
- Error rate scaling

**Key Results**:
- All 40+ tests passing
- Statistical equivalence within 10% tolerance
- Deterministic seeding verified: `config.seed -> worker_{id} -> noise + simulator`

### Phase 2: Parallel Coordinator (Done)
- [x] ParallelCoordinator structure
- [x] Worker pool management (WorkerState with per-worker World)
- [x] Result aggregation with deterministic ordering
- [x] ParallelConfig for flexible configuration
- [x] run() method for embarrassingly parallel execution
- [x] run_with_sync() method for rare event simulation with sync points
- [x] ExecutionStats for tracking progress
- [x] Comparison benchmarks vs MonteCarloRunner

### Phase 2.5: Performance Benchmarking (Done)

**Monte Carlo Comparison** (Bell state circuit, 4 workers):

| Shots | MonteCarloRunner | ParallelCoordinator | Ratio |
|-------|------------------|---------------------|-------|
| 100   | 80 µs (1.25 Melem/s) | 86 µs (1.17 Melem/s) | 1.07x slower |
| 1,000 | 348 µs (2.87 Melem/s) | 723 µs (1.38 Melem/s) | 2.1x slower |
| 10,000 | 2.47 ms (4.05 Melem/s) | 7.09 ms (1.41 Melem/s) | 2.9x slower |

**Key Insight**: ParallelCoordinator adds overhead from entity/component management.
Use MonteCarloRunner for simple Monte Carlo; ParallelCoordinator for rare event
simulation where entity splitting/sync is needed.

**Noise Application** (single-qubit gates):

| Gates | pecos-engines | pecos-neo | Ratio |
|-------|---------------|-----------|-------|
| 100   | 4.9 µs (20.5 Melem/s) | 36 µs (2.8 Melem/s) | 7.4x slower |
| 1,000 | 43 µs (23.3 Melem/s) | 295 µs (3.4 Melem/s) | 6.9x slower |
| 10,000 | 409 µs (24.4 Melem/s) | 2.9 ms (3.4 Melem/s) | 7.1x slower |

**Shot Execution** (100 shots, Bell state):

| Configuration | Time | Throughput |
|--------------|------|------------|
| pecos-engines (depol) | 51 µs | 1.95 Melem/s |
| pecos-neo (depol) | 79 µs | 1.26 Melem/s |
| pecos-neo (multi-channel) | 93 µs | 1.07 Melem/s |
| pecos-neo (no noise) | 37 µs | 2.68 Melem/s |

**CircuitRunner Reuse Impact** (100K iterations):

| Mode | Time | Per-iteration |
|------|------|---------------|
| New runner + noise each shot | 180 ms | 1.8 µs |
| New runner, no noise | 90 ms | 0.9 µs |
| Reused runner + noise | 80 ms | 0.8 µs |

**Key Finding**: Reusing runner is 2.25x faster than recreating per shot.

**Noise Channel Profiling** (1M iterations):

| Operation | Time/op | Notes |
|-----------|---------|-------|
| RNG sampling | 0.9 ns | Baseline |
| Single channel emit | 35.6 ns | 40x RNG |
| Three channels emit | 62.0 ns | 1.75x single |
| Noise model creation | 97.5 ns | Allocation overhead |

**Implemented Optimizations**:
1. Pre-sort handlers by priority when adding to noise model (avoids per-emit sort)
2. Precompute probability thresholds in `SingleQubitChannel` using `PecosRng::probability_threshold()`
3. Use `rng.check_probability(threshold)` instead of `rng.random::<f64>() < probability`

**Impact**: 5-9% improvement in noise channel performance, 17% faster new-runner creation.

**Remaining Optimization Opportunities**:
1. Apply probability threshold optimization to `TwoQubitChannel`, `MeasurementChannel`, etc.
2. Noise model creation is expensive - reuse where possible (2.25x speedup observed)
3. Event dispatch overhead - consider enum dispatch vs trait objects for hot paths

### Phase 3: Rare Event Support (Done)
- [x] Splitting criteria interface (`SplittingCriterion` trait, `ThresholdCriterion`, `CustomScoreCriterion`)
- [x] Synchronization points (`run_with_sync` in `ParallelCoordinator`)
- [x] Entity redistribution (`redistribute_by_weight`, `balance_entity_counts`, `EntityTransfer`)
- [x] Subset simulation support (demonstration in `tests/splitting_test.rs::test_subset_simulation_workflow`)

### Phase 4: Optimization (Complete)

#### Hot Path Profiling Results (benches/hot_path.rs)

**Noise Emission** (single operation, after optimizations):
| Operation | Time | Notes |
|-----------|------|-------|
| 1-qubit gate noise | ~20 ns | SingleQubitChannel |
| 2-qubit gate noise | ~18 ns | TwoQubitChannel |
| Measurement noise | ~16 ns | MeasurementChannel |

**Shot Execution** (single shot, after optimizations):
| Circuit | No Noise | With Noise | Overhead |
|---------|----------|------------|----------|
| Bell (2q, 4 gates) | 367 ns | 742 ns | 2.0x |
| 10-qubit (29 gates) | 3.1 µs | 5.2 µs | 1.7x |

**Multi-Shot Throughput**:
| Configuration | Time (100 shots) | Throughput |
|--------------|------------------|------------|
| Bell no noise | 43 µs | 2.3 Mshot/s |
| Bell with noise | 86 µs | 1.2 Mshot/s |

**Entity Operations**:
| Operation | Time | Notes |
|-----------|------|-------|
| Spawn entity (1q) | ~960 ns | Includes RNG seeding |
| Clone entity (10q) | 700 ns | |
| Split entity (10q, 4 copies) | 2.5 µs | |
| Resample 100 entities | 39 µs | ~390 ns/entity |

**Redistribution**:
| Configuration | Time | Per-entity |
|--------------|------|------------|
| 2w x 50e | 98 µs | ~1.0 µs |
| 4w x 50e | 194 µs | ~1.0 µs |
| 4w x 100e | 436 µs | ~1.1 µs |

**Simulator Operations**:
| Operation | Time |
|-----------|------|
| H gate | 12 ns |
| CX gate | 12 ns |
| Clone 10 qubits | 1.0 µs |
| Clone 50 qubits | 6.0 µs |
| Clone 100 qubits | 12.6 µs |

**RNG Operations**:
| Operation | Time |
|-----------|------|
| Seed from u64 | 7 ns |
| Random f64 | 0.9 ns |
| Probability check | 1.0 ns |

#### Key Findings

1. **Noise adds ~2x overhead** - Shot execution with noise takes roughly double the time
2. **Entity operations are fast** - Clone/split operations in the microsecond range
3. **Redistribution scales linearly** - ~1 µs per entity regardless of worker count
4. **Simulator clone dominates** - For large qubit counts, cloning the stabilizer tableau is the bottleneck

#### Optimization Status
- [x] Profile hot paths (benchmark suite created)
- [x] Optimize noise emission (see below)
- [x] Memory optimization (see below)
- [x] Simulator cloning analysis (see below)

#### Noise Emission Optimization (Completed)

Added `try_apply` method to `NoiseChannel` trait that combines `responds_to` + `apply` in one call,
avoiding redundant event matching. Implemented optimized versions in `SingleQubitChannel`,
`TwoQubitChannel`, and `MeasurementChannel`.

**Results**:
| Operation | Before | After | Improvement |
|-----------|--------|-------|-------------|
| 1-qubit gate noise | 34 ns | 29 ns | 14% faster |
| 2-qubit gate noise | 28 ns | 23 ns | 18% faster |
| Measurement noise | 29 ns | 21 ns | 28% faster |
| 10q circuit (with noise) | 5.9 µs | 5.6 µs | 5% faster |

#### Memory Optimization (Completed)

Analyzed type sizes and optimized `NoiseResponse` enum by boxing the `InjectGates` variant.

**Type Size Analysis**:
| Type | Size | Notes |
|------|------|-------|
| `QubitId` | 8 bytes | |
| `Angle64` | 8 bytes | |
| `GateCommand` | 72 bytes | Contains two SmallVecs |
| `SmallVec<[GateCommand; 4]>` | 296 bytes | Inline storage for 4 gates |
| `NoiseResponse` (before) | **304 bytes** | Dominated by InjectGates variant |
| `NoiseResponse` (after) | **48 bytes** | Boxing reduces to SmallVec<[QubitId; 4]> size |

**Optimization**: Changed `InjectGates(SmallVec<[GateCommand; 4]>)` to
`InjectGates(Box<SmallVec<[GateCommand; 4]>>)`. This trades a heap allocation for
much better cache efficiency in the hot noise emission path.

**Results**:
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| `NoiseResponse` size | 304 bytes | 48 bytes | **84% smaller** |
| Bell circuit (with noise) | 810 ns | 742 ns | **8.7% faster** |
| 10q circuit (with noise) | 5.6 µs | 5.1 µs | **9.9% faster** |

The smaller enum size improves cache locality during noise emission, more than
compensating for the Box allocation overhead.

#### Simulator Cloning Analysis (Completed)

Investigated lazy cloning (copy-on-write) vs reset optimization for simulators.

**Clone vs Reset Performance**:
| Qubits | Clone | Reset | Speedup |
|--------|-------|-------|---------|
| 10 | 964 ns | 121 ns | **8x faster** |
| 50 | 5.77 µs | 500 ns | **11.5x faster** |
| 100 | 11.9 µs | 1.03 µs | **11.6x faster** |

**Findings**:

1. **Lazy cloning (COW) is NOT beneficial** because:
   - Simulators ARE modified during shot execution (gates, measurements)
   - COW would trigger a copy on the first modification anyway
   - No sharing benefit since modifications happen immediately

2. **Reset-based optimization IS beneficial** for Monte Carlo:
   - Independent shots all start from |0⟩^n state
   - Reset is 8-12x faster than clone
   - Already used in pecos-engines' `MonteCarloEngine`
   - Added `run_shot_fresh()` to `ShotRunner` for this pattern

3. **For trajectory splitting** (rare events):
   - Clone IS necessary - each trajectory has meaningful state
   - Cannot use reset - branches must preserve their state
   - No optimization possible without major architectural changes

**Multi-Shot Benchmark (50 qubits, 100 shots)**:
| Pattern | Time | Notes |
|---------|------|-------|
| Clone-per-shot | 2.68 ms | Creates new simulator clone each shot |
| `run_shot_fresh` | 2.21 ms | Resets simulator between shots |
| Improvement | **21% faster** | |

**Recommendation**: Use `run_shot_fresh()` for Monte Carlo simulations where shots
are independent. Keep clone-per-entity for trajectory splitting where state must be
preserved.

#### MonteCarloRunner Validation (Completed)

The pecos-neo `MonteCarloRunner` has been validated against pecos-engines' `MonteCarloEngine`
to ensure statistical equivalence and measure performance differences.

**Validation Tests** (`tests/engine_comparison_test.rs`):
- Bell state distributions match (no noise)
- Depolarizing noise error rates match (within 10% tolerance)
- Measurement error rates match expected values
- Two-qubit noise decorrelation rates match
- Parallel execution produces consistent statistics
- Seed determinism verified across runs

All 8 validation tests pass, confirming that both implementations produce
statistically equivalent results.

**Performance Comparison** (benches/hot_path.rs, `monte_carlo_comparison` group):

QASM parsing is done in setup (outside timed section) for fair comparison.

| Benchmark | pecos-engines | pecos-neo | Speedup |
|-----------|---------------|-----------|---------|
| Bell (2q), 100 shots, no noise | 9.6 ms | 0.22 ms | **44x** |
| Bell (2q), 1000 shots, no noise | 64 ms | 1.1 ms | **58x** |
| Bell (2q), 1000 shots, with noise | 67 ms | 1.6 ms | **42x** |
| 10 qubits, 1000 shots, no noise | 293 ms | 4.1 ms | **71x** |

**Speedup Sources** (confirmed with QASM parsing excluded):
1. **No trait objects**: pecos-neo uses generics (zero-cost abstraction) vs `Box<dyn ...>`
2. **No ByteMessage serialization**: pecos-neo calls simulator methods directly
3. **Simpler architecture**: ShotRunner is much lighter than HybridEngine layers
4. **Direct command iteration**: No ControlEngine state machine overhead

## References

- Bevy ECS: https://bevyengine.org/learn/book/ecs/
- Data-Oriented Design: https://dataorienteddesign.com/dodbook/
- Mike Acton's DOD talk: https://www.youtube.com/watch?v=rX0ItVEVjHc
- Rare Event Simulation: Bucklew, "Introduction to Rare Event Simulation"
