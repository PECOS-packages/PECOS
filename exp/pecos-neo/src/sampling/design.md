# Advanced Sampling Architecture Design

This document explores the architecture for advanced sampling methods beyond
standard Monte Carlo, including importance sampling, splitting, subset simulation,
and branching program decomposition.

## Core Insight: Programs as Weighted Path Ensembles

A QEC program with classical control flow can be viewed as:
1. A directed graph of "blocks" (static circuit fragments)
2. Branch points where the path depends on measurement outcomes
3. Each complete path through the graph is a static circuit with a probability

```
┌─────────┐     ┌─────────┐     ┌─────────┐
│ Block 0 │────▶│ Branch  │────▶│Block 1A │──▶ ...
│  Prep   │     │ Decode  │     │ No corr │
└─────────┘     └────┬────┘     └─────────┘
                     │
                     │ correction needed
                     ▼
                ┌─────────┐
                │Block 1B │──▶ ...
                │ X corr  │
                └─────────┘
```

**Key observation**: We can sample paths and execute static circuits.

## Entity-Component Model

### Components (per simulation instance)

```rust
/// Core simulation state
struct SimulatorState {
    // The actual quantum state (e.g., stabilizer tableau)
    state: Box<dyn CliffordGateable>,
}

/// Noise context with O(1) lookups
struct NoiseContextComponent {
    ctx: NoiseContext,
}

/// Accumulated measurement outcomes
struct OutcomeHistory {
    outcomes: Vec<MeasurementOutcomes>,
}

/// Importance weight for this trajectory
struct ImportanceWeight {
    weight: SampleWeight,
}

/// Which path through the program this instance is following
struct PathState {
    current_block: BlockId,
    path_history: Vec<BranchChoice>,
}

/// Checkpoint for splitting methods
struct Checkpoint {
    saved_state: Option<Box<dyn CliffordGateable>>,
    saved_context: NoiseContext,
    saved_weight: SampleWeight,
}

/// Level for subset simulation
struct SubsetLevel {
    current_level: usize,
    level_achieved: bool,
}

/// Status of this simulation instance
enum SimulationStatus {
    Active,           // Still running
    Complete,         // Finished normally
    Failed,           // Uncorrectable error detected
    Pruned,           // Weight too low, removed
}
```

### Resources (shared across all instances)

```rust
/// The quantum program as a graph of blocks
struct ProgramGraph {
    blocks: Vec<CircuitBlock>,
    branches: Vec<BranchPoint>,
    initial_block: BlockId,
}

/// A static circuit block
struct CircuitBlock {
    id: BlockId,
    commands: CommandQueue,
    next: BlockTransition,
}

/// What happens after a block
enum BlockTransition {
    Continue(BlockId),           // Go to next block
    Branch(BranchId),            // Evaluate branch condition
    End,                         // Simulation complete
}

/// A branch point in the program
struct BranchPoint {
    id: BranchId,
    condition: BranchCondition,
    targets: Vec<(Predicate, BlockId)>,
}

/// Noise configuration for true distribution
struct TrueNoiseConfig {
    model: ComposableNoiseModel,
}

/// Noise configuration for proposal distribution (importance sampling)
struct ProposalNoiseConfig {
    model: ComposableNoiseModel,
}

/// Splitting configuration
struct SplittingConfig {
    importance_function: Box<dyn ImportanceFunction>,
    thresholds: Vec<f64>,
    max_clones: usize,
}

/// Subset simulation configuration
struct SubsetConfig {
    levels: Vec<LevelDefinition>,
    target_probability_per_level: f64,
}
```

## Systems (processing stages)

### Core Execution Systems

```rust
/// Execute a circuit block on all active instances
struct BlockExecutionSystem;

impl System for BlockExecutionSystem {
    fn run(&self, world: &mut World) {
        // Group instances by current block
        let by_block = world.group_by::<PathState, BlockId>();

        for (block_id, instances) in by_block {
            let block = world.resource::<ProgramGraph>().block(block_id);

            // Batch execute on all instances at this block
            for instance in instances {
                execute_block(instance, block, world);
            }
        }
    }
}

/// Apply noise to all active instances
struct NoiseSystem;

impl System for NoiseSystem {
    fn run(&self, world: &mut World) {
        let true_noise = world.resource::<TrueNoiseConfig>();
        let proposal_noise = world.resource::<ProposalNoiseConfig>();

        for instance in world.active_instances() {
            // Sample from proposal, compute weight
            let (response, weight_delta) = sample_with_importance(
                &event,
                true_noise,
                proposal_noise,
                instance.rng(),
            );

            // Update weight
            instance.get_mut::<ImportanceWeight>().update(weight_delta);

            // Apply response
            apply_noise_response(instance, response);
        }
    }
}

/// Evaluate branch conditions and update paths
struct BranchSystem;

impl System for BranchSystem {
    fn run(&self, world: &mut World) {
        let program = world.resource::<ProgramGraph>();

        for instance in world.instances_at_branch() {
            let branch = program.branch(instance.path().current_branch());
            let outcomes = instance.get::<OutcomeHistory>();

            // Evaluate which path to take
            let next_block = branch.evaluate(outcomes);

            // Update path state
            instance.get_mut::<PathState>().advance(next_block);
        }
    }
}
```

### Importance Sampling Systems

```rust
/// Update importance weights after noise application
struct ImportanceWeightSystem;

impl System for ImportanceWeightSystem {
    fn run(&self, world: &mut World) {
        // Weights are updated during NoiseSystem
        // This system can do post-processing like:
        // - Normalize weights
        // - Detect weight degeneracy
        // - Compute effective sample size
    }
}

/// Prune instances with negligible weight
struct WeightPruningSystem;

impl System for WeightPruningSystem {
    fn run(&self, world: &mut World) {
        let threshold = world.resource::<SamplingConfig>().prune_threshold;

        for instance in world.active_instances() {
            if instance.get::<ImportanceWeight>().is_negligible(threshold) {
                instance.set_status(SimulationStatus::Pruned);
            }
        }
    }
}
```

### Splitting Systems

```rust
/// Evaluate splitting criteria
struct SplitEvaluationSystem;

impl System for SplitEvaluationSystem {
    fn run(&self, world: &mut World) {
        let config = world.resource::<SplittingConfig>();

        for instance in world.active_instances() {
            let importance = config.importance_function.evaluate(instance);

            // Check if we crossed a threshold
            if let Some(clone_count) = config.should_split(importance) {
                // Mark for cloning
                instance.mark_for_cloning(clone_count);
            }
        }
    }
}

/// Clone marked instances
struct CloningSystem;

impl System for CloningSystem {
    fn run(&self, world: &mut World) {
        let to_clone: Vec<_> = world.instances_marked_for_cloning().collect();

        for (instance_id, clone_count) in to_clone {
            let instance = world.get(instance_id);

            // Split the weight among clones
            let split_weight = instance.get::<ImportanceWeight>().split(clone_count);

            // Create clones
            for _ in 0..clone_count {
                let clone = instance.deep_clone();
                clone.get_mut::<ImportanceWeight>().set(split_weight);
                world.spawn(clone);
            }
        }
    }
}
```

### Subset Simulation Systems

```rust
/// Check if instances have reached the next level
struct LevelCheckSystem;

impl System for LevelCheckSystem {
    fn run(&self, world: &mut World) {
        let config = world.resource::<SubsetConfig>();

        for instance in world.active_instances() {
            let level = instance.get::<SubsetLevel>().current_level;
            let outcomes = instance.get::<OutcomeHistory>();

            if config.levels[level + 1].is_satisfied(outcomes) {
                instance.get_mut::<SubsetLevel>().advance();
            }
        }
    }
}

/// Resample from instances that reached the current level
struct LevelResamplingSystem;

impl System for LevelResamplingSystem {
    fn run(&self, world: &mut World) {
        let config = world.resource::<SubsetConfig>();
        let target_count = config.instances_per_level;

        // Get instances at current level
        let at_level: Vec<_> = world.instances_at_level(current_level).collect();

        if at_level.len() >= target_count {
            // Resample: clone successful instances to maintain population
            resample_to_size(world, at_level, target_count);
        }
    }
}
```

### Statistics Collection

```rust
/// Collect weighted statistics from completed instances
struct StatisticsSystem;

impl System for StatisticsSystem {
    fn run(&self, world: &mut World) {
        let stats = world.resource_mut::<WeightedStatistics>();

        for instance in world.completed_instances() {
            let outcome = compute_outcome(instance); // e.g., logical error indicator
            let weight = instance.get::<ImportanceWeight>();

            stats.add(outcome, weight);
        }
    }
}
```

## Schedule / Execution Flow

```rust
/// Schedule for one "round" of simulation
struct SimulationSchedule {
    stages: Vec<Stage>,
}

enum Stage {
    // Core execution
    ExecuteBlocks,
    ApplyNoise,
    EvaluateBranches,

    // Sampling-specific
    UpdateWeights,
    PruneByWeight,
    EvaluateSplitting,
    PerformCloning,
    CheckLevels,
    Resample,

    // Collection
    CollectStatistics,
    CheckConvergence,
}

impl SimulationSchedule {
    /// Standard Monte Carlo schedule
    fn monte_carlo() -> Self {
        Self {
            stages: vec![
                Stage::ExecuteBlocks,
                Stage::ApplyNoise,
                Stage::EvaluateBranches,
                Stage::CollectStatistics,
            ],
        }
    }

    /// Importance sampling schedule
    fn importance_sampling() -> Self {
        Self {
            stages: vec![
                Stage::ExecuteBlocks,
                Stage::ApplyNoise,
                Stage::UpdateWeights,
                Stage::PruneByWeight,
                Stage::EvaluateBranches,
                Stage::CollectStatistics,
                Stage::CheckConvergence,
            ],
        }
    }

    /// Splitting schedule
    fn splitting() -> Self {
        Self {
            stages: vec![
                Stage::ExecuteBlocks,
                Stage::ApplyNoise,
                Stage::UpdateWeights,
                Stage::EvaluateSplitting,
                Stage::PerformCloning,
                Stage::EvaluateBranches,
                Stage::CollectStatistics,
            ],
        }
    }

    /// Subset simulation schedule
    fn subset_simulation() -> Self {
        Self {
            stages: vec![
                Stage::ExecuteBlocks,
                Stage::ApplyNoise,
                Stage::EvaluateBranches,
                Stage::CheckLevels,
                Stage::Resample,
                Stage::CollectStatistics,
            ],
        }
    }
}
```

## Branching Program Decomposition

For complex QEC programs with decoder feedback, we can:

1. **Analyze the program** to enumerate possible paths
2. **Compute path probabilities** (may need approximation for many branches)
3. **Sample paths** according to their probabilities
4. **Execute static circuits** for each sampled path
5. **Aggregate results** with appropriate weights

```rust
/// Decompose a branching program into weighted paths
fn decompose_program(program: &ProgramGraph) -> Vec<(Path, f64)> {
    let mut paths = Vec::new();
    enumerate_paths(program, program.initial_block, Path::new(), 1.0, &mut paths);
    paths
}

fn enumerate_paths(
    program: &ProgramGraph,
    block_id: BlockId,
    current_path: Path,
    probability: f64,
    paths: &mut Vec<(Path, f64)>,
) {
    let block = program.block(block_id);
    let extended_path = current_path.extend(block_id);

    match &block.next {
        BlockTransition::End => {
            paths.push((extended_path, probability));
        }
        BlockTransition::Continue(next) => {
            enumerate_paths(program, *next, extended_path, probability, paths);
        }
        BlockTransition::Branch(branch_id) => {
            let branch = program.branch(*branch_id);
            for (prob, target) in branch.outcome_probabilities() {
                enumerate_paths(program, target, extended_path.clone(), probability * prob, paths);
            }
        }
    }
}
```

## Open Questions

1. **Dynamic vs Static Scheduling**: Should the schedule adapt based on instance states,
   or be fixed upfront?

2. **Memory Management**: For 1M+ instances with splitting, memory becomes a concern.
   When should we checkpoint vs. re-simulate?

3. **Parallelization Strategy**:
   - Parallel across instances (current approach)
   - Parallel across blocks (when instances are grouped by path)
   - Hybrid

4. **Weight Degeneracy**: How to detect and handle when a few instances dominate?
   - Effective sample size monitoring
   - Adaptive resampling
   - Weight truncation

5. **Convergence Criteria**: When to stop sampling?
   - Target standard error
   - Minimum effective sample size
   - Maximum wall time

## Implementation Status

### Completed

1. **Sample Weight Tracking** (`weight.rs`)
   - `SampleWeight` with log-space storage for numerical stability
   - `WeightedStatistics` accumulator with proper normalization
   - `WeightedOutcome<T>` for associating weights with results

2. **Importance Sampling Configuration** (`importance.rs`)
   - `ImportanceConfig` with boost factor support
   - `ImportanceSamplingNoise` for tracking weights across error types
   - Methods for single-qubit, two-qubit, and measurement errors

3. **Importance Sampling Runner** (`importance_runner.rs`)
   - `ImportanceSamplingRunner<S>` that wraps a simulator
   - Samples from proposal distribution while tracking weights
   - Returns `ImportanceSampledShot` with outcomes and weight
   - Supports single-qubit, two-qubit, and measurement error boosts

4. **Monte Carlo Runner** (`monte_carlo.rs`)
   - `MonteCarloConfig` for configuring shots, workers, and seed
   - `MonteCarloRunner::run()` for parallel standard sampling
   - `MonteCarloRunner::run_importance()` for parallel importance sampling
   - `MonteCarloResults<T>` and `ImportanceSamplingResults` for results
   - Factory-function based design (no trait objects for runners)

5. **Classical-Quantum Hybrid Programs** (`program.rs`)
   - `CommandSource` trait for classical control generating quantum commands
   - `ProgramRunner<S>` for executing hybrid programs with measurement feedback
   - `StaticProgram` for single-batch circuits (no feedback)
   - `RepeatedProgram` for multiple rounds (e.g., QEC syndrome extraction)
   - `ConditionalProgram<F>` for branching based on measurement outcomes

6. **Classical Engine Adapter** (`adapter.rs`)
   - `ClassicalEngineAdapter<E>` wraps pecos-engines' `ClassicalControlEngine`
   - Implements `CommandSource` to bridge to pecos-neo infrastructure
   - Conversion utilities: `gate_to_command`, `gates_to_command_queue`, `command_queue_to_gates`
   - `ByteMessage` <-> `CommandQueue` / `MeasurementOutcomes` conversion
   - Always available as part of the core `pecos-neo` builder integration

### Next Steps

1. **Program Graph / Branching Programs**
   - Implement `ProgramGraph` for representing branching QEC programs
   - `CircuitBlock` for static circuit fragments
   - `BranchPoint` for measurement-dependent transitions
   - `BlockTransition` enum for control flow

2. **Entity-Component Storage**
   - Create a simple `World` struct with entity storage
   - Support for multiple simulation instances (for splitting)
   - Components: `SimulatorState`, `NoiseContext`, `ImportanceWeight`, `PathState`

3. **Splitting / Subset Simulation**
   - `SplitEvaluationSystem` to detect when trajectories should clone
   - `CloningSystem` to create copies with split weights
   - `LevelCheckSystem` for subset simulation levels

4. **Validation**
   - Compare importance sampling estimates against standard Monte Carlo
   - Validate against known QEC results (repetition code, surface code)
   - Benchmark variance reduction vs. computational cost
