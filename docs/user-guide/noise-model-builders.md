# Noise Model Builders

```hidden-python
from pecos import sim, Qasm, GeneralNoiseModelBuilder
```

PECOS provides builder classes for constructing quantum noise models with a fluent, method-chaining API. The `GeneralNoiseModelBuilder` is the most comprehensive builder, offering fine-grained control over various noise parameters.

## Quick Start

The simplest way to add noise to your QASM simulations is using the `GeneralNoiseModelBuilder`:

```python
from pecos import sim, Qasm, GeneralNoiseModelBuilder

# Define a circuit
qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
"""

# Create noise model with builder
noise = (
    GeneralNoiseModelBuilder()
    .with_seed(42)  # Reproducible randomness
    .with_p1(0.001)  # Single-qubit gate error
    .with_p2(0.01)
)  # Two-qubit gate error

# Use with sim()
results = sim(Qasm(qasm)).noise(noise).run(1000)
```

## GeneralNoiseModelBuilder

The `GeneralNoiseModelBuilder` provides methods to configure all aspects of quantum noise:

### Basic Error Probabilities

```python
noise = (
    GeneralNoiseModelBuilder()
    # Gate errors
    .with_p1(0.001)  # Single-qubit gate error
    .with_p2(0.01)  # Two-qubit gate error
    # State preparation and measurement
    .with_p_prep(0.0005)  # State preparation error
    .with_p_meas_0(0.002)  # Measurement 0→1 flip
    .with_p_meas_1(0.003)
)  # Measurement 1→0 flip
```

### Average vs Total Probabilities

The builder supports both "total" and "average" error probabilities:

```python
# Average probability (recommended for physical intuition)
noise = GeneralNoiseModelBuilder().with_average_p1(0.001).with_average_p2(0.01)  # Converted to total internally

# Total probability (used internally by the engine)
noise = GeneralNoiseModelBuilder().with_p1(0.00133).with_p2(0.0133)  # Total for single-qubit  # Total for two-qubit
```

**Note**: Average probabilities are more intuitive as they represent the actual error rate per gate. Total probabilities include a conversion factor based on the number of Pauli operators.

### Pauli Error Models

Specify custom Pauli error distributions instead of uniform depolarizing noise:

```python
noise = (
    GeneralNoiseModelBuilder()
    # Single-qubit Pauli errors
    .with_p1_pauli_model(
        {
            "X": 0.5,  # 50% X errors
            "Y": 0.3,  # 30% Y errors
            "Z": 0.2,  # 20% Z errors
        }
    )
    # Two-qubit Pauli errors
    .with_p2_pauli_model(
        {
            "IX": 0.25,  # 25% error on second qubit only
            "XI": 0.25,  # 25% error on first qubit only
            "XX": 0.5,  # 50% correlated X errors
        }
    )
)
```

### Scaling and Global Parameters

```python
noise = (
    GeneralNoiseModelBuilder()
    .with_seed(42)  # Random seed for reproducibility
    .with_scale(1.5)  # Scale all error rates by 1.5x
    .with_leakage_scale(0.1)  # 10% of errors cause leakage
    .with_emission_scale(0.05)
)  # 5% spontaneous emission
```

### Noiseless Gates

Make specific gates ideal (no noise):

```python
noise = (
    GeneralNoiseModelBuilder()
    .with_p1(0.001)
    .with_p2(0.01)
    # Single gate
    .with_noiseless_gate("H")
    # Multiple gates
    .with_noiseless_gate("S")
    .with_noiseless_gate("T")
    .with_noiseless_gate("MEASURE")
)
```

### Idle Locations

`Idle` gates are timing markers by default. They do not silently inherit
single-qubit gate noise from `p1` or `with_p1(...)`.

Configure idle decoherence with any combination of these independent families:

- `with_p_idle_linear(rate, model)` samples one linear-rate event from a
  normalized X/Y/Z/L distribution.
- `with_p_idle_sin_squared(rate, model)` independently samples each X/Y/Z/L
  mechanism with `sin²(rate * multiplier * duration)`. Its rate is radians per
  time unit and its multipliers are intentionally unnormalized because each
  axis has its own rate.
- `with_p_idle_coherent(rate, model)` deterministically applies RX/RY/RZ with
  angle `rate * multiplier * duration`. Its rate is radians per time unit, with
  no `2*pi` or coherent-to-incoherent conversion. Its model is also
  intentionally unnormalized: the values are relative generator-rate
  multipliers, not probabilities. Omitting the Python model uses
  `{"RX": 1.0, "RY": 1.0, "RZ": 1.0}`.

The legacy `with_p_idle_quadratic_rate(...)` path remains available.
`with_p_idle_quadratic_coherent(...)` selects whether that one legacy rate emits
coherent RZ rotations or a stochastic sine-squared Z twirl; it does not control
the independent coherent family. Combining that switch set to `True` with the
coherent family is rejected because both would emit idle rotations.

Coherent evolution is not sampled and consumes no RNG draws. Whether it can be
consumed depends on the downstream consumer: the standard DEM builder rejects
coherent idle noise, the EEG route in `exp/pecos-eeg` represents it with an RZ
generator, and a simulator applies it only when it has a rotation executor.
PECOS #437 tracks the case where a missing executor silently dropped rotations.

To add the same kind of idle-noise site to both qubits after every two-qubit
gate, set its duration with `with_idle_after_2q(...)`:

```python
noise = GeneralNoiseModelBuilder().with_p_idle_linear_rate(0.01).with_idle_after_2q(1.0)
```

The duration only chooses where and how long idling occurs. It is not a
standalone probability: all configured linear, sine-squared, coherent, and
legacy quadratic idle mechanisms apply at these sites just as they do at a
scheduled `Idle` gate. A duration of `0.0` disables the after-two-qubit sites.
Consequently, code that previously used `with_p2_idle(0.01)` without a linear
idle rate now produces no after-2q idle noise; the equivalent configuration is
`with_p_idle_linear_rate(0.01).with_idle_after_2q(1.0)`.

## Common Noise Model Examples

### Basic Depolarizing Noise

Simple uniform noise on all operations:

```python
# Uniform depolarizing noise
noise = (
    GeneralNoiseModelBuilder().with_p1(0.001).with_p2(0.01).with_p_prep(0.001).with_p_meas_0(0.001).with_p_meas_1(0.001)
)
```

### Realistic Hardware Noise

Model based on typical superconducting qubit parameters:

```python
noise = (
    GeneralNoiseModelBuilder()
    .with_seed(42)
    # Gate errors (two-qubit gates are typically 10x worse)
    .with_average_p1(0.0001)  # 0.01% single-qubit error
    .with_average_p2(0.001)  # 0.1% two-qubit error
    # State prep and measurement (often dominant errors)
    .with_p_prep(0.001)  # 0.1% prep error
    .with_p_meas_0(0.01)  # 1% false positive
    .with_p_meas_1(0.005)
)  # 0.5% false negative
```

### Ion Trap Noise Model

Ion traps have different characteristics than superconducting qubits:

```python
noise = (
    GeneralNoiseModelBuilder()
    .with_seed(42)
    # Excellent single-qubit gates
    .with_average_p1(0.00001)  # 0.001% error
    # Two-qubit gates are the limiting factor
    .with_average_p2(0.003)  # 0.3% error
    # State preparation
    .with_p_prep(0.001)  # 0.1% error
    # Asymmetric measurement (bright/dark state detection)
    .with_p_meas_0(0.001)  # Dark state error
    .with_p_meas_1(0.005)
)  # Bright state error (higher)
```

### Biased Noise Model

Model with biased errors (e.g., more phase errors than bit flips):

```python
noise = (
    GeneralNoiseModelBuilder()
    # Biased single-qubit errors
    .with_average_p1(0.001)
    .with_p1_pauli_model(
        {
            "X": 0.1,  # 10% bit flips
            "Y": 0.1,  # 10% Y errors
            "Z": 0.8,  # 80% phase errors (dominant)
        }
    )
    # Biased two-qubit errors
    .with_average_p2(0.01)
    .with_p2_pauli_model(
        {
            "IZ": 0.3,  # 30% phase on second qubit
            "ZI": 0.3,  # 30% phase on first qubit
            "ZZ": 0.2,  # 20% correlated phase
            "XX": 0.2,  # 20% other errors
        }
    )
)
```


## Complete Example

Here's a comprehensive example showing various builder features:

```python
from pecos import sim, Qasm
from pecos_rslib import GeneralNoiseModelBuilder
from collections import Counter

# QASM circuit: 3-qubit GHZ state
qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
cx q[0], q[1];
cx q[1], q[2];
measure q -> c;
"""

# Build comprehensive noise model
noise = (
    GeneralNoiseModelBuilder()
    # Reproducibility
    .with_seed(42)
    # Global scaling
    .with_scale(1.2)  # 20% higher error rates
    # Make Hadamard gates perfect
    .with_noiseless_gate("H")
    # State preparation
    .with_p_prep(0.001)
    # Single-qubit gates with biased errors
    .with_average_p1(0.0001)
    .with_p1_pauli_model(
        {
            "X": 0.2,
            "Y": 0.2,
            "Z": 0.6,  # More dephasing
        }
    )
    # Two-qubit gates
    .with_average_p2(0.001)
    # Asymmetric measurement
    .with_p_meas_0(0.002)
    .with_p_meas_1(0.005)
)

# Run simulation
results = sim(Qasm(qasm)).noise(noise).run(1000)

# Analyze results
counts = Counter(results.to_dict()["c"])
print("GHZ state measurement results:")
for state, count in counts.most_common(5):
    binary = format(state, "03b")
    print(f"|{binary}>: {count}")
```

## Tips

1. **Use Average Probabilities**: They're more intuitive and match experimental error rates.

2. **Set Seeds for Reproducibility**: Always use `.with_seed()` for reproducible results in research.

3. **Start Simple**: Begin with uniform probabilities, then add complexity as needed.

4. **Match Hardware Specs**: Use error rates from device calibration data when available.

5. **Consider Error Hierarchies**: Typically: measurement > two-qubit > state prep > single-qubit.

6. **Use Noiseless Gates Sparingly**: Only for gates that are effectively perfect (e.g., virtual Z rotations).

## Comparison with Predefined Noise Models

While builders offer maximum flexibility, PECOS also provides simpler convenience functions:

```python
from pecos import depolarizing_noise, GeneralNoiseModelBuilder

# Simple depolarizing (uniform probability)
simple = depolarizing_noise().with_uniform_probability(0.001)

# Equivalent with GeneralNoiseModelBuilder
builder = (
    GeneralNoiseModelBuilder()
    .with_p1(0.001)
    .with_p2(0.001)
    .with_p_prep(0.001)
    .with_p_meas_0(0.001)
    .with_p_meas_1(0.001)
)

# Builder advantages:
# - Fine-grained control
# - Pauli error models
# - Scaling factors
# - Noiseless gates
# - Crosstalk modeling
```

## Next Steps

- For performance optimization, see [QASM Simulation Guide](qasm-simulation.md)
- For the complete API reference, see the [API Documentation](../api/api-reference.md)
