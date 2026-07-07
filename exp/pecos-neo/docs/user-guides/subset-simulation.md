# Subset Simulation

Estimate extremely small failure probabilities (10^-6 or below) that are too
rare for standard Monte Carlo or importance sampling.

## How It Works

Instead of sampling the rare event directly, decompose it into a chain of more
likely events:

```
P(failure) = P(F1) x P(F2|F1) x P(F3|F2) x ... x P(Fm|Fm-1)
```

Each conditional probability is ~0.1, so you only need ~1000 samples per level
instead of 1/P total samples.

## Quick Start

Subset simulation is a sampling strategy on `sim_neo()`, like
`monte_carlo()` and `importance_sampling()`. It needs two functions:
a score (how close is this outcome to failure?) and a failure predicate.
Both are required; the result arrives in `results.subset` (per-shot
`outcomes` are empty for subset runs).

```rust
use pecos_neo::tool::{sim_neo, sparse_stab, subset_simulation};

let results = sim_neo(circuit)
    .quantum(sparse_stab())   // currently the only supported backend
    .noise(noise)
    .sampling(
        subset_simulation(1000)  // samples per level
            // Higher score = closer to failure
            .score(|outcomes| count_syndrome_errors(outcomes) as f64)
            // What counts as failure?
            .failure(|outcomes| is_logical_error(outcomes)),
    )
    .seed(42)
    .run();

let subset = results.subset.expect("subset strategy returns an estimate");
println!("P(failure) = {:.2e}", subset.probability());
println!("95% CI: {:?}", subset.confidence_interval_95());
```

Requires a static circuit on the `sparse_stab()` backend; checked at
`.build()`.

## Configuration

```rust
subset_simulation(1000)             // Samples per level (more = slower but more precise)
    .score(score_fn)                // Required: distance-to-failure metric
    .failure(failure_fn)            // Required: rare event predicate
    .threshold_fraction(0.1)        // Top 10% advance each level
    .max_levels(20)                 // Safety limit on levels
    .min_conditional_prob(1e-6)     // Give up below this conditional probability
```

## Direct library API

For lower-level control (custom noise factories, checkpoint-continuation
variants like `ProperSubsetSimulation`), use the sampling module directly:

```rust
use pecos_neo::sampling::subset::{SubsetSimulation, SubsetConfig};

let config = SubsetConfig::new()
    .with_samples_per_level(1000)
    .with_seed(42);

let result = SubsetSimulation::new(circuit, num_qubits, score_fn, is_failure)
    .with_noise_builder(|| Some(noise.clone()))
    .with_config(config)
    .run();
```

## When to Use

- **Below 10^-6**: Subset simulation is designed for this regime
- **10^-3 to 10^-6**: [Importance sampling](importance-sampling.md) is usually simpler
- **Above 10^-3**: Standard Monte Carlo is probably fine

## Tips

- The score function is critical -- it must correlate with failure
- Start with `samples_per_level(500)` for debugging, scale up for final runs
- Monitor `coefficient_of_variation` -- values below 0.3 indicate good precision

## Going Deeper

For `ProperSubsetSimulation`, `QecSubsetSimulation`, ECS-based trajectory
management, score function design, and validation tools, see the
[developer subset simulation guide](../dev/subset-simulation.md).
