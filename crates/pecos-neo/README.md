# pecos-neo

Composable quantum simulation with event-driven noise modeling.

This crate provides a composable approach to quantum simulation with:

- **Typed Commands**: `GateCommand` and `CommandQueue` replacing `ByteMessage`
- **Composable Noise**: Event-driven noise channels that can be freely combined
- **Plugin System**: Bevy-inspired architecture for bundling functionality
- **Simple Runner**: Direct simulator execution via `ShotRunner`

## Architecture

The key insight is **composition over configuration**. Instead of a monolithic noise model with dozens of parameters, you compose small, focused channels:

```
┌─────────────────────────────────────────────────────────────┐
│                   ComposableNoiseModel                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│  │ SingleQubit │ │  TwoQubit   │ │ Measurement │  ...       │
│  │   Channel   │ │   Channel   │ │   Channel   │            │
│  └─────────────┘ └─────────────┘ └─────────────┘            │
│         │               │               │                    │
│         └───────────────┴───────────────┘                    │
│                         │                                    │
│                    NoiseEvent                                │
│              (BeforeGate, AfterGate,                         │
│               AfterMeasurement, etc.)                        │
└─────────────────────────────────────────────────────────────┘
```

## Usage Patterns

### 1. Direct Composition (Most Flexible)

Compose exactly the channels you need:

```rust
use pecos_neo::prelude::*;
use pecos_neo::noise::*;
use pecos_neo::noise::plugins::CorePlugin;

let noise = ComposableNoiseModel::new()
    .add_plugin(CorePlugin)  // State tracking
    .add_channel(SingleQubitChannel::depolarizing(0.001))
    .add_channel(TwoQubitChannel::depolarizing(0.01)
        .with_angle_scaling(AngleScaling::linear()))
    .add_channel(MeasurementChannel::asymmetric(0.02, 0.03))
    .add_channel(IdleChannel::linear(0.0001)
        .with_linear_depolarizing());
```

### 2. Convenience Builders (Familiar API)

Use builders that mirror existing noise models:

```rust
use pecos_neo::noise::GeneralNoiseModelBuilder;

let noise = GeneralNoiseModelBuilder::new()
    .with_p1(0.001)
    .with_p2(0.01)
    .with_p_meas(0.02, 0.03)
    .with_p_prep(0.005)
    .with_p_idle_linear(0.0001)
    .build();
```

### 3. Mixed Approach (Best of Both)

Start with a builder, then customize:

```rust
use pecos_neo::noise::*;

// Start with standard configuration
let noise = GeneralNoiseModelBuilder::new()
    .with_p1(0.001)
    .with_p2(0.01)
    .build()
    // Add custom channels
    .add_channel(CrosstalkChannel::new()
        .with_global_rate(0.001)
        .with_transitions(CrosstalkTransitions::symmetric_with_leakage()));
```

### 4. Custom Channels

Implement your own noise channels:

```rust
use pecos_neo::noise::*;
use pecos_rng::PecosRng;

#[derive(Clone)]
struct MyCustomChannel {
    error_rate: f64,
}

impl NoiseChannel for MyCustomChannel {
    fn responds_to(&self, event: &NoiseEvent<'_>) -> bool {
        matches!(event, NoiseEvent::AfterGate { .. })
    }

    fn apply(
        &self,
        event: &NoiseEvent<'_>,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> NoiseResponse {
        // Your custom noise logic here
        NoiseResponse::None
    }

    fn name(&self) -> &'static str {
        "MyCustomChannel"
    }

    fn clone_box(&self) -> Box<dyn NoiseChannel> {
        Box::new(self.clone())
    }
}
```

## Running Simulations

```rust
use pecos_neo::prelude::*;
use pecos_qsim::SparseStab;

// Build a circuit
let commands = CommandBuilder::new()
    .pz(0)
    .pz(1)
    .h(0)
    .cx(0, 1)
    .mz(0)
    .mz(1)
    .build();

// Create a runner with noise
let mut runner = ShotRunner::new(SparseStab::new(2))
    .with_noise(noise)
    .with_seed(42);

// Run multiple shots
for _ in 0..1000 {
    let outcomes = runner.run_shot(&commands);
    println!("{}", outcomes.bitstring());
}
```

## Universal Simulation (Rotation Gates)

For simulators that support arbitrary rotation gates (state vector simulators),
use `execute_all()` instead of `execute()`:

```rust
use pecos_neo::prelude::*;
use pecos_qsim::StateVec;

let commands = CommandBuilder::new()
    .pz(0)
    .rx(0, Angle64::HALF_TURN)  // RX(pi) = X gate
    .rzz(0, 1, Angle64::QUARTER_TURN)  // RZZ(pi/2)
    .mz(0)
    .build();

let mut runner = ShotRunner::new(StateVec::new(2));
let outcomes = runner.execute_all(&commands);
```

Supported rotation gates: RX, RY, RZ, T, Tdg, U, R1XY, RXX, RYY, RZZ, CRZ, CCX (Toffoli).

CCX and CRZ are automatically decomposed into supported gates.

## Idle Time Modeling

The `IdleChannel` models T1/T2 decay during idle periods.
Time is specified in abstract time units - the interpretation (nanoseconds, clock cycles, etc.)
is defined by the noise model configuration.

```rust
use pecos_neo::prelude::*;

let commands = CommandBuilder::new()
    .pz(0)
    .idle(0, 100u64)  // 100 time units
    .mz(0)
    .build();

let noise = ComposableNoiseModel::new()
    .add_channel(IdleChannel::linear(0.001)  // 0.1% error per time unit
        .with_linear_depolarizing());        // X/Y/Z with equal probability
```

### Physical Time Units

For physicists who prefer working with physical times (nanoseconds, microseconds),
configure the time scale at the model level:

```rust
use pecos_neo::prelude::*;
use pecos_core::TimeScale;

// Define: 1 TimeUnit = 1 nanosecond, then use physical times
let noise = ComposableNoiseModel::new()
    .with_time_scale(TimeScale::NANOSECONDS)
    .with_idle_t1_t2(50e-6, 30e-6);  // T1=50us, T2=30us in seconds
```

Available time scales: `NANOSECONDS`, `MICROSECONDS`, `MILLISECONDS`, `SECONDS`,
or custom via `TimeScale::from_cycle_time_ns(50.0)` for gate-cycle-based timing.

You can also add precision to coarse units:
```rust
// Think in seconds, but with nanosecond precision (9 decimal places)
let scale = TimeScale::SECONDS.with_precision(9);
// Now 0.00005 seconds = 50,000 TimeUnits
```

## Available Channels

| Channel | Purpose |
|---------|---------|
| `SingleQubitChannel` | Pauli/emission errors on 1Q gates |
| `TwoQubitChannel` | Pauli/emission errors on 2Q gates, angle-dependent scaling |
| `MeasurementChannel` | Asymmetric readout errors |
| `PreparationChannel` | Initialization errors and leakage |
| `IdleChannel` | T1/T2 decay, coherent/incoherent dephasing |
| `LeakageChannel` | Handles effects of leaked qubits |
| `CrosstalkChannel` | Local/global crosstalk with state-dependent transitions |

## Plugins

Plugins bundle related functionality:

```rust
use pecos_neo::noise::plugins::*;

let noise = ComposableNoiseModel::new()
    .add_plugin(CorePlugin)                           // State tracking
    .add_plugin(LeakagePlugin::new())                 // Leakage handling
    .add_plugin(DepolarizingPlugin::new(0.01, 0.02)); // Simple depolarizing
```

## Key Types

### Pauli Weights

Control error distributions:

```rust
// Uniform (default)
PauliWeights::uniform()  // 1/3 X, 1/3 Y, 1/3 Z

// Z-biased (dephasing)
PauliWeights::z_biased(0.9)  // 90% Z, 5% X, 5% Y

// Custom
PauliWeights::custom(0.1, 0.2, 0.7)  // 10% X, 20% Y, 70% Z
```

### Emission Weights

Control leakage vs Pauli errors:

```rust
// Pauli only (no leakage)
SingleQubitEmissionWeights::uniform()

// Include leakage
SingleQubitEmissionWeights::uniform_with_leakage()  // 25% each X/Y/Z/leak

// Leakage only
SingleQubitEmissionWeights::leakage_only()
```

### Angle Scaling

For parameterized gates (RZZ, etc.):

```rust
// No angle dependence
AngleScaling::constant()

// Linear: error ~ |theta/pi|
AngleScaling::linear()

// Quadratic: error ~ (theta/pi)^2
AngleScaling::quadratic()

// Full polynomial: a + b*|theta/pi| + c*|theta/pi|^d
// (matches GeneralNoiseModel's p2_angle_* parameters)
AngleScaling::polynomial(a, b, c, d)

// Asymmetric: different scaling for +/- angles
// offset + linear*|theta/pi| + scale*|theta/pi|^power
AngleScaling::asymmetric(
    neg_offset, neg_linear, neg_scale,
    pos_offset, pos_linear, pos_scale,
    power
)
```

### Crosstalk Transitions

State-dependent crosstalk effects:

```rust
// Simple flip model
CrosstalkTransitions::flip_only()

// Include leakage
CrosstalkTransitions::symmetric_with_leakage()

// Custom per-state transitions
CrosstalkTransitions::custom(
    from_0_stay, from_0_flip, from_0_leak,
    from_1_stay, from_1_flip, from_1_leak,
)
```

## Large-Scale Simulations (1M+ Qubits)

`NoiseContext` uses bit vectors internally for O(1) qubit state lookups,
enabling efficient simulation of large qubit counts:

```rust
use pecos_neo::noise::NoiseContext;

// Pre-allocate for large simulations (optional but recommended)
let ctx = NoiseContext::with_capacity(1_000_000);
```

Performance (measured on typical hardware):

| Qubits | Time |
|--------|------|
| 1K     | 0.09 ms |
| 10K    | 0.8 ms  |
| 100K   | 70 ms   |
| 1M     | 1.0 s   |

See the `large_scale` example for a benchmark.

## Design Philosophy

1. **Composition over Configuration**: Build complex models from simple pieces
2. **Explicit over Implicit**: See exactly what channels are active
3. **Flexible Foundation**: The composable system supports any noise model
4. **Convenience Wrappers**: Builders provide familiar APIs for common patterns
5. **Extensible**: Easy to add custom channels without modifying core code
6. **Scalable**: DOD context enables million-qubit stabilizer simulations
