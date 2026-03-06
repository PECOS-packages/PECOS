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

```rust
use pecos_neo::sampling::subset::{SubsetSimulation, SubsetConfig};

// Define: what does "closer to failure" mean?
let score_fn = |outcomes: &MeasurementOutcomes| -> f64 {
    // Higher score = closer to failure
    count_syndrome_errors(outcomes) as f64
};

// Define: what counts as failure?
let is_failure = |outcomes: &MeasurementOutcomes| -> bool {
    is_logical_error(outcomes)
};

let config = SubsetConfig::new()
    .with_samples_per_level(1000)
    .with_seed(42);

let result = SubsetSimulation::new(circuit, num_qubits, score_fn, is_failure)
    .with_noise_builder(|| Some(noise.clone()))
    .with_config(config)
    .run();

println!("P(failure) = {:.2e}", result.probability());
println!("95% CI: [{:.2e}, {:.2e}]", result.confidence_interval_95());
```

## Configuration

```rust
SubsetConfig::new()
    .with_samples_per_level(1000)   // Samples per level (more = slower but more precise)
    .with_threshold_fraction(0.1)   // Top 10% advance each level
    .with_max_levels(20)            // Safety limit on levels
    .with_seed(42)                  // For reproducibility
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
