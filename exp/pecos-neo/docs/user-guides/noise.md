# Adding Noise

## Quickest Option

One-liner on `sim_neo`:

```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
sim_neo(circuit).auto().depolarizing(0.01)?.sampling(monte_carlo(1000)).run();
# Ok(())
# }
```

## Per-Channel Control

Set different rates for single-qubit, two-qubit, and measurement errors:

```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
sim_neo(circuit).auto()
    .noise(GeneralNoiseModelBuilder::new()
        .with_p1(0.001)
        .with_p2(0.01)
        .with_p_meas_symmetric(0.005).build()?)
    .sampling(monte_carlo(1000))
    .run();
# Ok(())
# }
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
crosstalk, and more. It and the presets below require the `composite-noise`
Cargo feature on `pecos-neo`:

```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
use pecos_neo::noise::composite::CompositeNoiseModelBuilder;
use pecos_core::QubitId;

// Two-qubit device: each qubit is the other's neighbor.
fn neighbors(active: &[QubitId]) -> Vec<QubitId> {
    [QubitId(0), QubitId(1)].into_iter().filter(|q| !active.contains(q)).collect()
}

let noise = CompositeNoiseModelBuilder::new()
    // Gate errors
    .with_p1(0.001)?                     // Single-qubit error rate
    .with_p2(0.01)?                      // Two-qubit error rate
    .with_p1_emission_ratio(0.1)?        // Fraction of errors that leak (vs Pauli)

    // Measurement
    .with_p_meas(0.02, 0.03)?           // Asymmetric: P(report 1|true 0), P(report 0|true 1)

    // Preparation
    .with_p_prep(0.001)?                // Prep error rate

    // Leakage
    .with_leakage(0.001, 0.1)?         // Leak rate, seepage rate

    // Crosstalk
    .with_p_meas_crosstalk(0.001, 0.01)?  // Global, local rates
    .with_p_meas_crosstalk_local_fn(neighbors)

    .build()?;
# Ok(())
# }
```

## Common Presets

**Ion trap style:**
```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
CompositeNoiseModelBuilder::new()
    .with_p1(0.0001)?.with_p2(0.001)?
    .with_p1_emission_ratio(0.1)?.with_p2_emission_ratio(0.1)?
    .with_leakage(0.0001, 0.1)?
    .with_p_meas(0.001, 0.01)?
    .build()?;
# Ok(())
# }
```

**Superconducting style:**
```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
CompositeNoiseModelBuilder::new()
    .with_p1(0.001)?.with_p2(0.01)?
    .with_idle_t1_t2(50e-6, 30e-6)?
    .with_p_meas_symmetric(0.02)?
    .build()?;
# Ok(())
# }
```

**Pure depolarizing (testing):**
```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
CompositeNoiseModelBuilder::new().with_p1(p)?.with_p2(p)?.build()?;
# Ok(())
# }
```

## Going Deeper

For custom noise primitives, implementing your own `NoiseChannel`, composable
decision trees, geometric sampling, and full API reference, see the
[developer noise guide](../dev/noise.md).
