# Subset Simulation for Rare Event Estimation

Subset simulation is a multilevel Monte Carlo method for efficiently estimating very small probabilities (10^-6 or smaller). This is particularly useful for quantum error correction where logical error rates can be extremely low.

## The Problem

Standard Monte Carlo estimation of P(rare event) requires ~1/P samples for reasonable accuracy. For P = 10^-10, this means 10^10 samples - computationally infeasible.

## The Solution: Subset Simulation

Decompose the rare event into a sequence of more frequent intermediate events:

```
P(F) = P(F₁) × P(F₂|F₁) × P(F₃|F₂) × ... × P(Fₘ|Fₘ₋₁)
```

Each conditional probability is ~0.1-0.2, requiring only ~1000 samples per level. Total samples: ~1000 × m instead of 1/P.

## Algorithm Overview

1. **Define a score function** that increases as the system approaches failure
2. **Run initial samples** to completion, recording scores
3. **Set adaptive threshold** at the (1-p₀) quantile of scores
4. **Resample**: Keep samples above threshold, clone them to replace samples below
5. **Continue** from cloned states with fresh randomness
6. **Repeat** until failure threshold is reached
7. **Multiply** conditional probabilities for final estimate

## API Overview

pecos-neo provides several subset simulation implementations:

| Struct | Use Case |
|--------|----------|
| `SubsetSimulation` | General-purpose, works with any circuit and noise model |
| `ProperSubsetSimulation` | Full Au & Beck algorithm with checkpoint history |
| `BernoulliSubsetSimulation` | Validation tool using synthetic Bernoulli process |
| `QecSubsetSimulation` | QEC-specific with synthetic syndrome generation |
| `EcsSubsetSimulation` | ECS-based trajectory management |

## SubsetSimulation (Recommended)

### Basic Usage

```rust
use pecos_neo::sampling::subset::{SubsetSimulation, SubsetConfig};
use pecos_neo::command::CommandBuilder;
use pecos_neo::noise::composite::CompositeNoiseModelBuilder;
use pecos_neo::outcome::MeasurementOutcomes;
use pecos_core::QubitId;

// Build a circuit
let circuit = CommandBuilder::new()
    .pz(0)
    .pz(1)
    .h(0)
    .cx(0, 1)
    .mz(0)
    .mz(1)
    .build();

// Define score function: higher score = closer to failure
// Example: count of qubits measuring 1
let score_fn = |outcomes: &MeasurementOutcomes| -> f64 {
    let q0 = outcomes.get_bit(QubitId(0)).unwrap_or(false);
    let q1 = outcomes.get_bit(QubitId(1)).unwrap_or(false);
    (q0 as u8 + q1 as u8) as f64
};

// Define failure predicate
let is_failure_fn = |outcomes: &MeasurementOutcomes| -> bool {
    // Fail if both qubits measure 1
    outcomes.get_bit(QubitId(0)).unwrap_or(false) &&
    outcomes.get_bit(QubitId(1)).unwrap_or(false)
};

// Configure
let config = SubsetConfig::new()
    .with_samples_per_level(1000)
    .with_threshold_fraction(0.1)  // Top 10% advance each level
    .with_max_levels(20)
    .with_seed(42);

// Run with noise
let noise_builder = || {
    Some(CompositeNoiseModelBuilder::new()
        .with_p1(0.01)
        .with_p2(0.05)
        .build())
};

let result = SubsetSimulation::new(circuit, 2, score_fn, is_failure_fn)
    .with_noise_builder(noise_builder)
    .with_config(config)
    .run();

println!("P(failure) = {:.2e}", result.probability());
println!("Coefficient of variation: {:.2}", result.coefficient_of_variation);
println!("Total samples: {}", result.total_samples);
```

### SubsetConfig Options

```rust
SubsetConfig::new()
    // Number of samples to maintain at each level
    .with_samples_per_level(1000)

    // Fraction of samples that exceed threshold (determines threshold adaptively)
    // Lower = more levels but better precision
    .with_threshold_fraction(0.1)

    // Maximum levels before stopping
    .with_max_levels(20)

    // Minimum conditional probability before declaring failure unreachable
    .with_min_conditional_prob(1e-6)

    // Random seed for reproducibility
    .with_seed(42)
```

### SubsetResult

```rust
pub struct SubsetResult {
    /// Statistics for each level
    pub levels: Vec<LevelStats>,

    /// Final probability estimate
    pub probability: f64,

    /// Coefficient of variation (relative uncertainty)
    pub coefficient_of_variation: f64,

    /// Total samples across all levels
    pub total_samples: usize,

    /// Number of direct failures observed
    pub direct_failures: usize,
}

// Get confidence interval (approximate)
let (lower, upper) = result.confidence_interval_95();
println!("95% CI: [{:.2e}, {:.2e}]", lower, upper);
```

## ProperSubsetSimulation

Implements the full Au & Beck algorithm with checkpoint history for proper trajectory restart:

```rust
use pecos_neo::sampling::subset::ProperSubsetSimulation;

let sim = ProperSubsetSimulation::new(
    0.1,    // p_damage: probability of "damage" per round
    1.0,    // damage_increment: amount of damage per event
    20,     // num_rounds: number of rounds to simulate
    5.0,    // failure_threshold: fail if damage >= this
    SubsetConfig::new()
        .with_samples_per_level(1000)
        .with_seed(42),
);

let result = sim.run();
```

This implementation:
- Stores full trajectory history with checkpoints
- Properly restarts from the exact point where trajectories crossed thresholds
- Provides unbiased estimates even for correlated processes

## BernoulliSubsetSimulation

A validation tool that uses a simple Bernoulli damage accumulation model with known analytical solution:

```rust
use pecos_neo::sampling::subset::BernoulliSubsetSimulation;

let sim = BernoulliSubsetSimulation::new(
    0.1,    // p_damage: probability of damage per step
    50,     // num_steps: number of steps
    10.0,   // failure_threshold: fail if total damage >= this
)
.with_config(SubsetConfig::new()
    .with_samples_per_level(5000)
    .with_seed(42));

// Get analytical probability for comparison
let analytical = sim.analytical_probability();
println!("Analytical P(failure): {:.6}", analytical);

// Run Monte Carlo (note: BernoulliSubsetSimulation::run() does direct MC)
let result = sim.run();
println!("MC estimate: {:.6}", result.probability());

// Verify they match
let error = (analytical - result.probability()).abs() / analytical;
assert!(error < 0.1, "Relative error too large: {}", error);
```

**Note**: `BernoulliSubsetSimulation::run()` uses direct Monte Carlo for validation purposes. For actual subset simulation of Bernoulli processes, use `ProperSubsetSimulation`.

## QecSubsetSimulation

Specialized for quantum error correction with synthetic syndrome generation:

```rust
use pecos_neo::sampling::subset::{QecSubsetSimulation, QecSubsetConfig};
use pecos_neo::ecs::World;
use pecos_qsim::SparseStab;
use pecos_core::QubitId;

// Configure QEC simulation
let config = QecSubsetConfig::new(
    20,                                  // num_rounds: QEC cycles
    vec![QubitId(3), QubitId(4)],        // ancilla_qubits
    6.0,                                 // failure_threshold: syndrome weight
)
.with_samples_per_level(500)
.with_threshold_fraction(0.1)
.with_seed(42);

// Create world with trajectories
let mut world: World<SparseStab> = World::new(42);
for _ in 0..500 {
    world.spawn_with_simulator(SparseStab::new(5));  // 3 data + 2 ancilla
}

// Run with synthetic syndromes
let sim = QecSubsetSimulation::new(world, config);
let result = sim.run_proper(0.2);  // p_syndrome = 0.2 per ancilla per round

println!("P(logical_failure) = {:.2e}", result.probability());
```

**Design Note**: `QecSubsetSimulation` uses synthetic syndrome generation (random Bernoulli trials) rather than running actual noisy quantum circuits. This allows:
- Validation against known binomial statistics
- Fast iteration during algorithm development
- Isolated testing of the subset simulation algorithm

For production QEC simulations with actual noise, use `SubsetSimulation` with `ComposableNoiseModel` via `with_noise_builder()`.

## Choosing Score Functions

The score function critically affects efficiency. Good score functions:

1. **Correlate with failure**: Higher scores should indicate higher failure probability
2. **Vary smoothly**: Avoid discrete jumps that create poor thresholds
3. **Are cheap to compute**: Called for every sample

### Examples

**Syndrome weight (QEC)**:
```rust
let score_fn = |outcomes: &MeasurementOutcomes| -> f64 {
    ancilla_qubits.iter()
        .filter(|&&q| outcomes.get_bit(q).unwrap_or(false))
        .count() as f64
};
```

**Error count**:
```rust
let score_fn = |outcomes: &MeasurementOutcomes, errors: &ErrorHistory| -> f64 {
    errors.total_count() as f64
};
```

**Decoder confidence**:
```rust
let score_fn = |outcomes: &MeasurementOutcomes| -> f64 {
    let (decoded, confidence) = decoder.decode(outcomes);
    1.0 - confidence  // Lower confidence = higher score
};
```

## ECS-Based Trajectory Management

For advanced use cases, `EcsSubsetSimulation` uses an Entity Component System for trajectory management:

```rust
use pecos_neo::ecs::{World, SplitDecision};
use pecos_neo::sampling::subset::EcsSubsetSimulation;

// World manages entities with:
// - Simulator state (cloneable)
// - Sample weight
// - Trajectory metadata

let mut world: World<SparseStab> = World::new(seed);

// Spawn trajectories
for _ in 0..num_samples {
    world.spawn_with_simulator(SparseStab::new(num_qubits));
}

// Apply split decisions (prune, keep, or clone)
let decisions: Vec<(EntityId, usize)> = compute_splits(&world, threshold);
world.apply_split_decisions(&decisions);

// Resample by weight
world.resample_by_weight(target_count, &mut rng);
```

## Helper Circuits

Convenience functions for common QEC circuits:

```rust
use pecos_neo::sampling::subset::{bit_flip_syndrome_circuit, phase_flip_syndrome_circuit};

// 3-qubit bit-flip code syndrome extraction
// Data: q0, q1, q2  |  Ancilla: q3, q4
// Measures Z0Z1 (q3) and Z1Z2 (q4)
let circuit = bit_flip_syndrome_circuit();

// 3-qubit phase-flip code syndrome extraction
// Measures X0X1 (q3) and X1X2 (q4)
let circuit = phase_flip_syndrome_circuit();
```

## Performance Tips

1. **Start with small samples**: Debug with 100-500 samples, scale up for final runs

2. **Tune threshold_fraction**:
   - 0.1 (10%) is typical
   - Lower values = more levels but better precision per level
   - Higher values = fewer levels but more variance

3. **Monitor coefficient of variation**: CV < 0.3 indicates good precision

4. **Use seeded runs**: For debugging and reproducibility

5. **Profile score function**: It's called O(samples × levels) times

## References

- Au, S.K. and Beck, J.L. (2001). "Estimation of small failure probabilities in high dimensions by subset simulation." Probabilistic Engineering Mechanics.

- Cérou, F., et al. (2012). "Sequential Monte Carlo for rare event estimation." Statistics and Computing.
