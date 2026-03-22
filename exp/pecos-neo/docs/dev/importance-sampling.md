# Importance Sampling

Importance sampling is a variance reduction technique for Monte Carlo estimation. By sampling from a biased distribution and reweighting, we can estimate rare event probabilities more efficiently than direct sampling.

## Overview

Standard Monte Carlo estimates E[f(X)] by averaging f(X) over samples from P(X).

Importance sampling instead:
1. Samples from a biased distribution Q(X) that emphasizes important regions
2. Reweights samples by the likelihood ratio P(X)/Q(X)
3. The weighted average is an unbiased estimator with (potentially) lower variance

## Quick Start with sim_neo

The easiest way to use importance sampling is via the `sim_neo` Tool API with the `importance_sampling()` builder:

```rust
use pecos_neo::tool::{sim_neo, importance_sampling};
use pecos_neo::command::CommandBuilder;
use pecos_core::QubitId;

let circuit = CommandBuilder::new()
    .pz(0).pz(1)
    .h(0).cx(0, 1)
    .mz(0).mz(1)
    .build();

// Run with importance sampling
let results = sim_neo(circuit)
    .orchestrator(importance_sampling()
        .with_p1(0.001)      // Single-qubit error rate
        .with_p2(0.01)       // Two-qubit error rate
        .with_p_meas(0.001)  // Measurement error rate
        .with_boost(10.0))   // Boost factor
    .shots(10000)
    .seed(42)
    .run();

// Compute weighted statistics
if let Some(error_rate) = results.weighted_mean(|outcome| {
    // Your failure indicator function
    if check_logical_error(outcome) { 1.0 } else { 0.0 }
}) {
    println!("Estimated error rate: {:.2e}", error_rate);
}
```

For uniform error rates, use the `with_uniform_error()` method:

```rust
let results = sim_neo(circuit)
    .orchestrator(importance_sampling()
        .with_uniform_error(0.001)  // Same rate for all gate types
        .with_boost(10.0))
    .shots(10000)
    .run();
```

The results include importance weights that can be used with `weighted_mean()` or `weighted_stats()` to compute unbiased estimates.

## API Components

### ImportanceSamplingRunner

The main runner for importance-sampled circuit execution:

```rust
use pecos_neo::sampling::{ImportanceSamplingRunner, OutcomeBiasConfig};
use pecos_neo::noise::composite::CompositeNoiseModelBuilder;
use pecos_simulators::SparseStab;

// Configure outcome biasing
let bias_config = OutcomeBiasConfig::new()
    .with_target_outcome(true)   // Bias toward measuring 1
    .with_bias_strength(0.8);    // 80% bias (vs 50% unbiased)

// Create runner with biased sampling
let noise = CompositeNoiseModelBuilder::new()
    .with_p1(0.001)
    .build();

let mut runner = ImportanceSamplingRunner::new(SparseStab::new(num_qubits))
    .with_noise(noise)
    .with_outcome_bias(bias_config);

// Run circuit
runner.run(&circuit);

// Get weighted result
let shot = runner.take_shot();
println!("Outcome: {:?}", shot.outcomes);
println!("Weight: {}", shot.weight.weight());  // Likelihood ratio
```

### OutcomeBiasConfig

Configure how measurement outcomes are biased:

```rust
OutcomeBiasConfig::new()
    // Target outcome to bias toward
    .with_target_outcome(true)  // Bias toward 1

    // Strength of bias (0.5 = unbiased, 1.0 = always target)
    .with_bias_strength(0.8)

    // Apply only to specific qubits
    .with_target_qubits(vec![QubitId(0), QubitId(1)])

    // Boost factor for rare events (multiplies effective bias)
    .with_boost(2.0)
```

### ImportanceSampledShot

Result of an importance-sampled execution:

```rust
pub struct ImportanceSampledShot {
    /// The measurement outcomes
    pub outcomes: MeasurementOutcomes,

    /// Importance weight (likelihood ratio)
    pub weight: SampleWeight,

    /// Number of measurements that were biased
    pub biased_count: usize,
}
```

### SampleWeight

Tracks importance weights in log-space for numerical stability:

```rust
use pecos_neo::sampling::SampleWeight;

// Create weight
let w = SampleWeight::from_linear(0.5);

// Combine weights (multiply)
let combined = w1.combine(w2);

// Get linear weight
let linear = w.weight();

// Get log weight
let log_w = w.log_weight();
```

## Use Cases

### Estimating Rare Measurement Outcomes

```rust
// Want to estimate P(all qubits measure 1) for a circuit
// where this is rare (e.g., P ~ 10^-6)

let bias_config = OutcomeBiasConfig::new()
    .with_target_outcome(true)
    .with_bias_strength(0.9);  // Strong bias toward 1

let mut sum_weights = 0.0;
let mut count = 0;

for _ in 0..num_samples {
    runner.run(&circuit);
    let shot = runner.take_shot();

    // Check if this is our target event
    let all_ones = shot.outcomes.as_slice()
        .iter()
        .all(|o| o.outcome);

    if all_ones {
        sum_weights += shot.weight.weight();
        count += 1;
    }

    runner.reset();
}

// Weighted estimate
let p_estimate = sum_weights / num_samples as f64;
println!("P(all 1s) ~ {:.2e} (from {} biased hits)", p_estimate, count);
```

### Combining with Noise

Importance sampling can be combined with noise models:

```rust
let noise = CompositeNoiseModelBuilder::new()
    .with_p1(0.001)
    .with_p2(0.01)
    .build();

let runner = ImportanceSamplingRunner::new(SparseStab::new(num_qubits))
    .with_noise(noise)
    .with_outcome_bias(bias_config);
```

### Weighted Statistics

Use `WeightedStatistics` to accumulate importance-weighted results:

```rust
use pecos_neo::sampling::weight::WeightedStatistics;

let mut stats = WeightedStatistics::new();

for _ in 0..num_samples {
    runner.run(&circuit);
    let shot = runner.take_shot();

    // Add weighted outcome
    let value = compute_value(&shot.outcomes);
    stats.add_weighted(value, shot.weight);

    runner.reset();
}

println!("Weighted mean: {}", stats.weighted_mean());
println!("Effective sample size: {}", stats.effective_sample_size());
```

## ImportanceSamplingChannel

Wrap any noise channel to add importance sampling:

```rust
use pecos_neo::sampling::importance::ImportanceSamplingChannel;

// Wrap a channel to track importance weights
let wrapped = ImportanceSamplingChannel::new(
    original_channel,
    ImportanceConfig::new()
        .with_boost(2.0),
);
```

## Theory

### Likelihood Ratio

For a measurement with true probability p of outcome 1:
- Unbiased: sample with probability p
- Biased: sample with probability q (e.g., q = 0.9)

The likelihood ratio is:
- If outcome = 1: w = p/q
- If outcome = 0: w = (1-p)/(1-q)

The weighted sample is an unbiased estimator: E_Q[w × f(X)] = E_P[f(X)]

### Variance Reduction

Importance sampling reduces variance when Q emphasizes regions where f(X) is large. For indicator functions (rare events), biasing toward the rare event dramatically reduces variance.

Optimal bias: Q(X) ∝ |f(X)| × P(X)

### Effective Sample Size

The effective sample size accounts for weight variability:

ESS = (Σ wᵢ)² / Σ wᵢ²

Low ESS indicates high weight variance and potential issues.

## Best Practices

1. **Match bias to target**: Bias toward outcomes you're trying to estimate

2. **Monitor effective sample size**: ESS << N indicates problems

3. **Don't overbias**: Very strong bias (>0.95) can cause high weight variance

4. **Combine with stratification**: For multi-level rare events, consider subset simulation

5. **Validate on known cases**: Test bias configuration on circuits with known probabilities

## Limitations

- Requires knowing which outcomes to bias toward
- Weight variance can grow exponentially with circuit depth
- Not suitable for all rare event types (subset simulation may be better)

## References

- Rubinstein, R.Y. and Kroese, D.P. (2016). "Simulation and the Monte Carlo Method." Wiley.

- Owen, A.B. (2013). "Monte Carlo theory, methods and examples." (Online book)
