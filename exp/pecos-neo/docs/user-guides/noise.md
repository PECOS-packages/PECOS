# Adding Noise

## Quickest Option

One-liner on `sim_neo`:

```rust
sim_neo(circuit).auto().depolarizing(0.01).sampling(monte_carlo(1000)).run();
```

## Per-Channel Control

Set different rates for single-qubit, two-qubit, and measurement errors:

```rust
sim_neo(circuit).auto()
    .noise(GeneralNoiseModelBuilder::new()
        .with_p1(0.001)
        .with_p2(0.01)
        .with_p_meas_symmetric(0.005))
    .sampling(monte_carlo(1000))
    .run();
```

## How It Works

Under the hood, noise is event-driven. As the simulator runs your circuit, it
emits events -- `AfterGate`, `AfterMeasurement`, `IdleTime`, etc. Each noise
channel declares which events it responds to and injects errors when they fire.
The builders above set all of this up for you.

When you need more control -- picking individual channels, reacting to custom
signals, or writing your own -- see [Composable Noise Model](noise-model.md)
and [Noise Channels](noise-channels.md).

## Full Noise Model Builder

`CompositeNoiseModelBuilder` gives you control over leakage, emission, seepage,
crosstalk, and more:

```rust
use pecos_neo::noise::composite::CompositeNoiseModelBuilder;

let noise = CompositeNoiseModelBuilder::new()
    // Gate errors
    .with_p1(0.001)                     // Single-qubit error rate
    .with_p2(0.01)                      // Two-qubit error rate
    .with_p1_emission_ratio(0.1)        // Fraction of errors that leak (vs Pauli)

    // Measurement
    .with_p_meas(0.02, 0.03)           // Asymmetric: P(report 1|true 0), P(report 0|true 1)

    // Preparation
    .with_p_prep(0.001)                // Prep error rate

    // Leakage
    .with_leakage(0.001, 0.1)         // Leak rate, seepage rate

    // Crosstalk
    .with_measurement_crosstalk(0.001, 0.01)  // Global, local rates

    .build();
```

## Common Presets

**Ion trap style:**
```rust
CompositeNoiseModelBuilder::new()
    .with_p1(0.0001).with_p2(0.001)
    .with_p1_emission_ratio(0.1).with_p2_emission_ratio(0.1)
    .with_leakage(0.0001, 0.1)
    .with_p_meas(0.001, 0.01)
    .build();
```

**Superconducting style:**
```rust
CompositeNoiseModelBuilder::new()
    .with_p1(0.001).with_p2(0.01)
    .with_idle_t1_t2(50e-6, 30e-6)
    .with_p_meas_symmetric(0.02)
    .build();
```

**Pure depolarizing (testing):**
```rust
CompositeNoiseModelBuilder::new().with_p1(p).with_p2(p).build();
```

## Going Deeper

For custom noise primitives, implementing your own `NoiseChannel`, composable
decision trees, geometric sampling, and full API reference, see the
[developer noise guide](../dev/noise.md).
