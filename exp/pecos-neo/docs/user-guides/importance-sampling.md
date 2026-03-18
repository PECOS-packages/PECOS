# Importance Sampling

Estimate rare event probabilities (10^-3 to 10^-6) by boosting error rates and
reweighting the results. Much more efficient than brute-force Monte Carlo.

## Quick Start

```rust
use pecos_neo::tool::{sim_neo, importance_sampling};

let results = sim_neo(circuit)
    .orchestrator(importance_sampling()
        .with_p1(0.001)      // Single-qubit error rate
        .with_p2(0.01)       // Two-qubit error rate
        .with_boost(10.0))   // Sample 10x more errors
    .shots(10000)
    .seed(42)
    .run();
```

For uniform rates across all gate types:

```rust
sim_neo(circuit)
    .orchestrator(importance_sampling()
        .with_uniform_error(0.001)
        .with_boost(10.0))
    .shots(10000)
    .run();
```

## Reading Results

Results include importance weights. Use `weighted_mean()` to get unbiased estimates:

```rust
if let Some(error_rate) = results.weighted_mean(|outcome| {
    if check_logical_error(outcome) { 1.0 } else { 0.0 }
}) {
    println!("Logical error rate: {:.2e}", error_rate);
}
```

## When to Use

- **10^-3 to 10^-6**: Importance sampling works well here
- **Below 10^-6**: Consider [subset simulation](subset-simulation.md) instead
- **Above 10^-3**: Standard Monte Carlo is probably fine

## Tips

- Start with `boost(5.0)` and increase if effective sample size is too low
- Don't overbias (boost > 20) -- weight variance can explode
- Validate on circuits with known error rates first

## Going Deeper

For the `ImportanceSamplingRunner` API, `SampleWeight` internals, outcome
biasing, and the math behind likelihood ratios, see the
[developer importance sampling guide](../dev/importance-sampling.md).
