# Architecture Evolution: DOD/ECS Patterns in PECOS

This document captures observations about the data-oriented design (DOD) and ECS-inspired patterns in pecos-neo, and how they relate to the broader PECOS architecture.

## Two Equivalent Approaches

pecos-neo provides the same back-and-forth control flow as pecos-engines, but using DOD/functional patterns instead of OOP/state-machine patterns.

| pecos-engines | pecos-neo |
|---------------|-----------|
| `ControlEngine::start()` | `CommandSource::next_commands(None)` |
| `ControlEngine::continue_processing(measurements)` | `CommandSource::next_commands(Some(outcomes))` |
| `EngineStage::NeedsProcessing(commands)` | `Some(commands)` |
| `EngineStage::Complete(result)` | `None` |
| `EngineSystem::process_as_system()` loop | `ProgramRunner::run_shot()` loop |
| `MonteCarloEngine` | `CircuitRunner` + `World<S>` |
| Existing classical engines | `ClassicalEngineAdapter` wraps them |

## pecos-engines (Classic OOP/Trait Pattern)

The existing engine system uses a controller pattern with trait objects:

```rust
trait ControlEngine {
    fn start(&mut self, input) -> EngineStage<EngineInput, Output>;
    fn continue_processing(&mut self, result) -> EngineStage<EngineInput, Output>;
}

enum EngineStage<I, O> {
    NeedsProcessing(I),
    Complete(O),
}
```

Classical engines (QASM, HUGR, PHIR, PhirJson) implement this trait, managing state internally and communicating with quantum engines via `ByteMessage`.

### pecos-neo (DOD/Functional Pattern)

pecos-neo uses a simpler, more functional approach:

```rust
trait CommandSource {
    fn next_commands(&mut self, outcomes: Option<&MeasurementOutcomes>) -> Option<CommandQueue>;
    fn is_complete(&self) -> bool;
    fn reset(&mut self);
}
```

Key differences:
- Single method instead of two-phase protocol
- Data flows as parameters (outcomes in, commands out)
- `Option<CommandQueue>` instead of enum variants

### ECS Infrastructure

pecos-neo includes lightweight ECS-inspired infrastructure for population-based simulation:

```rust
struct World<S: CliffordGateable> {
    // Entity management
    alive_entities: BTreeSet<EntityId>,

    // Component storage (Structure of Arrays)
    simulators: ComponentStorage<SimulatorComponent<S>>,
    rngs: ComponentStorage<RngComponent>,
    weights: ComponentStorage<WeightComponent>,
    noise_contexts: ComponentStorage<NoiseContextComponent>,
    outcomes: ComponentStorage<OutcomeComponent>,
    statuses: ComponentStorage<StatusComponent>,

    // Shared resources
    resources: Resources,
}
```

This supports:
- Trajectory splitting/cloning for rare event simulation
- Weight-based resampling for subset simulation
- Deterministic seeding per entity
- Parallel execution across entities

## Bridging the Two Worlds

The `ClassicalEngineAdapter` bridges existing engines to the new pattern:

```rust
// Wrap any existing classical engine
let engine = QASMEngine::from_str(qasm)?;
let mut program = ClassicalEngineAdapter::new(engine);

// Use with pecos-neo infrastructure
let mut runner = ProgramRunner::new(SparseStab::new(2))
    .with_noise(ComposableNoiseModel::new()
        .add_channel(SingleQubitChannel::depolarizing(0.01)));

let result = runner.run_shot(&mut program);
```

## Evolution Path

### Phase 1: Prove Patterns in pecos-neo (Current)

- `World<S>` with component storage for population simulation
- `CommandSource` / `ProgramRunner` for control flow
- `ComposableNoiseModel` with event-driven noise channels
- `DecompositionRegistry` with O(1) gate lookup
- `BatchedCircuit` for cache-friendly execution
- Plugin system for extensible gate definitions
- Adapters to bridge existing pecos-engines code

### Phase 2: Identify Wins

Measure and validate:
- Which patterns provide performance gains
- Where cache locality matters most
- What parallelism opportunities exist
- How the API ergonomics compare

### Phase 3: Refactor Deeper Components

**Simulators** could adopt SoA layouts:
```rust
// Current (Array of Structs)
struct SparseStab {
    tableau: Vec<StabilizerRow>,
}

// DOD (Struct of Arrays)
struct SparseStabDOD {
    x_bits: BitMatrix,      // cache-friendly
    z_bits: BitMatrix,      // SIMD-friendly
    phases: Vec<u8>,
}
```

**Classical Engines** could use systems/events:
```rust
fn classical_control_system(
    mut commands: EventWriter<CommandBatch>,
    measurements: EventReader<MeasurementsReady>,
    mut program_state: ResMut<ProgramState>,
) { ... }
```

## Expected Optimization Opportunities

| Pattern | Benefit |
|---------|---------|
| SoA memory layout | Cache-friendly iteration, SIMD vectorization |
| Batched operations | Amortize overhead, enable parallelism |
| Event-driven flow | Decouple components, async-friendly |
| System scheduling | Auto-parallelize independent work |
| Change detection | Skip unchanged entities |
| Component queries | Process only relevant subsets |
| Dense array indexing | O(1) lookup by GateId instead of HashMap |

## Concrete Use Cases

1. **Population-based simulation** - World<S> already enables rare event methods
2. **Multi-shot Monte Carlo** - Batch across shots with shared resources
3. **Large circuits** - BatchedCircuit groups operations by type
4. **Noise injection** - Parallel noise sampling per entity
5. **Gate decomposition** - Plugin-based, resolved at build time

## Non-Linear Program Support

PECOS handles multiple program representations with varying complexity:

| Format | Control Flow |
|--------|-------------|
| QASM 2.0 | `if (creg == val) op` only |
| PhirJson | `if/else` blocks, `sequence`, `qparallel` |
| PHIR | MLIR-style regions/blocks with terminators |
| HUGR | Full CFG, Conditional, TailLoop, FuncDefn/Call |

The gate system and DOD patterns operate orthogonally to control flow - gates are decomposed and executed regardless of how the program's control flow is structured.

## Design Principles

1. **Data as parameters** - Outcomes passed in, commands returned
2. **Composition over inheritance** - Plugins, channels, adapters
3. **Dense storage** - Arrays indexed by ID, not HashMaps
4. **Batch-friendly** - Group similar operations together
5. **Deterministic** - BTreeMap/BTreeSet for ordered iteration, derived seeds
