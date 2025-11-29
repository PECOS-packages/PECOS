# Noise Model Configuration

The `GeneralNoiseFactory` provides a flexible, dictionary-based approach to creating quantum noise models in PECOS. It maps configuration keys to `GeneralNoiseModelBuilder` methods, enabling JSON/dict-based noise model creation with type safety and validation.

## Overview

The factory pattern allows you to:
- Create noise models from JSON files or Python dictionaries
- Use predefined parameter mappings or define custom ones
- Validate configurations before applying them
- Maintain backward compatibility while adding new features

## Basic Usage

### Using Default Factory

```python
from pecos.rslib import GeneralNoiseFactory, qasm_sim

# Create factory with default mappings
factory = GeneralNoiseFactory()

# Define noise configuration
config = {
    "seed": 42,
    "p1": 0.001,  # Single-qubit gate error
    "p2": 0.01,  # Two-qubit gate error
    "p_meas_0": 0.002,  # Measurement 0->1 flip
    "p_meas_1": 0.003,  # Measurement 1->0 flip
}

# Create noise model
noise = factory.create_from_dict(config)

# Use in simulation
results = qasm_sim(qasm).noise(noise).run(1000)
```

### From JSON Configuration

```python
import json
from pecos.rslib import create_noise_from_json

# JSON configuration
json_config = """
{
    "seed": 42,
    "p1": 0.001,
    "p2": 0.01,
    "scale": 1.5,
    "noiseless_gates": ["H", "MEASURE"],
    "p1_pauli_model": {
        "X": 0.5,
        "Y": 0.3,
        "Z": 0.2
    }
}
"""

# Create noise model directly
noise = create_noise_from_json(json_config)
```

## Predefined Parameter Mappings

The default factory includes 43 predefined mappings:

| Configuration Key | Builder Method | Description |
|------------------|----------------|-------------|
| **Global Parameters** |||
| `seed` | `with_seed` | Random seed for reproducibility |
| `scale` | `with_scale` | Global error rate scaling factor |
| `leakage_scale` | `with_leakage_scale` | Leakage vs depolarizing ratio (0-1) |
| `emission_scale` | `with_emission_scale` | Spontaneous emission scaling |
| `seepage_prob` | `with_seepage_prob` | Global seepage probability for leaked qubits |
| `noiseless_gate` | `with_noiseless_gate` | Single gate to make noiseless |
| `noiseless_gates` | `with_noiseless_gate` | List of gates to make noiseless |
| **Idle Noise** |||
| `p_idle_coherent` | `with_p_idle_coherent` | Use coherent vs incoherent dephasing |
| `p_idle_linear_rate` | `with_p_idle_linear_rate` | Idle noise linear rate |
| `p_idle_average_linear_rate` | `with_p_average_idle_linear_rate` | Average idle noise linear rate |
| `p_idle_linear_model` | `with_p_idle_linear_model` | Idle noise Pauli distribution |
| `p_idle_quadratic_rate` | `with_p_idle_quadratic_rate` | Idle noise quadratic rate |
| `p_idle_average_quadratic_rate` | `with_p_average_idle_quadratic_rate` | Average idle noise quadratic rate |
| `p_idle_coherent_to_incoherent_factor` | `with_p_idle_coherent_to_incoherent_factor` | Coherent to incoherent conversion |
| `idle_scale` | `with_idle_scale` | Idle noise scaling factor |
| **State Preparation** |||
| `p_prep` | `with_prep_probability` | State preparation error probability |
| `p_prep_leak_ratio` | `with_prep_leak_ratio` | Fraction of prep errors that leak |
| `p_prep_crosstalk` | `with_p_prep_crosstalk` | Preparation crosstalk probability |
| `prep_scale` | `with_prep_scale` | Preparation error scaling factor |
| `p_prep_crosstalk_scale` | `with_p_prep_crosstalk_scale` | Preparation crosstalk scaling |
| **Single-Qubit Gates** |||
| `p1` | `with_p1_probability` | Single-qubit gate error probability |
| `p1_average` | `with_average_p1_probability` | Average single-qubit error |
| `p1_emission_ratio` | `with_p1_emission_ratio` | Fraction that are emission errors |
| `p1_emission_model` | `with_p1_emission_model` | Single-qubit emission distribution |
| `p1_seepage_prob` | `with_p1_seepage_prob` | Probability of seeping leaked qubits |
| `p1_pauli_model` | `with_p1_pauli_model` | Pauli error distribution |
| `p1_scale` | `with_p1_scale` | Single-qubit error scaling factor |
| **Two-Qubit Gates** |||
| `p2` | `with_p2_probability` | Two-qubit gate error probability |
| `p2_average` | `with_average_p2_probability` | Average two-qubit error |
| `p2_angle_params` | `with_p2_angle_params` | RZZ angle-dependent error params |
| `p2_angle_power` | `with_p2_angle_power` | Power parameter for angle errors |
| `p2_emission_ratio` | `with_p2_emission_ratio` | Fraction that are emission errors |
| `p2_emission_model` | `with_p2_emission_model` | Two-qubit emission distribution |
| `p2_seepage_prob` | `with_p2_seepage_prob` | Probability of seeping leaked qubits |
| `p2_pauli_model` | `with_p2_pauli_model` | Pauli error distribution |
| `p2_idle` | `with_p2_idle` | Idle noise after two-qubit gates |
| `p2_scale` | `with_p2_scale` | Two-qubit error scaling factor |
| **Measurement** |||
| `p_meas` | `with_meas_probability` | Symmetric measurement error |
| `p_meas_0` | `with_meas_0_probability` | Probability of 0->1 flip |
| `p_meas_1` | `with_meas_1_probability` | Probability of 1->0 flip |
| `p_meas_crosstalk` | `with_p_meas_crosstalk` | Measurement crosstalk probability |
| `meas_scale` | `with_meas_scale` | Measurement error scaling |
| `p_meas_crosstalk_scale` | `with_p_meas_crosstalk_scale` | Measurement crosstalk scaling |

## Safety Features

### Override Warnings

The factory warns when you override default mappings:

```python
factory = GeneralNoiseFactory()
factory.add_mapping("p1", "with_p2_probability", float)
# UserWarning: Overriding default mapping for 'p1':
# 'with_p1_probability' -> 'with_p2_probability'
```

### Mapping Visualization

View current mappings with visual indicators:

```python
factory.show_mappings()
# Current Parameter Mappings:
# ================================================================================
# Configuration Key    → Builder Method                     Description
# --------------------------------------------------------------------------------
# *p1                 → with_p2_probability                Single-qubit gate error
#  p2                 → with_p2_probability                Two-qubit gate error
# ...
# * = Overridden default mapping
```

### Strict Mode Validation

By default, unknown keys raise errors:

```python
# Strict mode (default)
factory.create_from_dict({"unknown_key": 123})  # Raises ValueError

# Non-strict mode
factory.create_from_dict({"unknown_key": 123}, strict=False)  # Ignores unknown keys
```

## Customization

### Starting Without Defaults

Create an empty factory for complete control:

```python
# Three equivalent ways to create an empty factory
factory = GeneralNoiseFactory(use_defaults=False)
factory = GeneralNoiseFactory.empty()
```

### Adding Custom Mappings

```python
factory = GeneralNoiseFactory.empty()

# Add custom mappings with domain-specific terminology
factory.add_mapping(
    "single_gate_error",
    "with_p1_probability",
    float,
    "Error rate for single-qubit gates",
)
factory.add_mapping(
    "two_gate_error", "with_p2_probability", float, "Error rate for two-qubit gates"
)
factory.add_mapping(
    "readout_error", "with_meas_0_probability", float, "Readout error probability"
)

# Use domain-specific configuration
config = {
    "single_gate_error": 0.001,
    "two_gate_error": 0.01,
    "readout_error": 0.002,
}
noise = factory.create_from_dict(config)
```

### Removing Unwanted Mappings

Remove mappings you don't need:

```python
factory = GeneralNoiseFactory()

# Remove mappings if not needed
factory.remove_mapping("p1_average")  # Remove average probability
factory.remove_mapping("p2_average")  # Remove average probability
```

### Custom Type Converters

Add mappings with custom value conversion:

```python
# Convert percentage to probability
def percent_to_prob(percent):
    return percent / 100.0


factory.add_mapping(
    "p1_percent", "with_p1_probability", percent_to_prob, "P1 error as percentage"
)

# Use percentage in config
config = {"p1_percent": 0.1}  # 0.1% = 0.001 probability
```

## Advanced Examples

### Ion Trap Noise Model

Create a specialized factory for ion trap quantum computers:

```python
# Create ion trap specific factory
factory = GeneralNoiseFactory.empty()

# Add ion trap terminology
factory.add_mapping(
    "state_prep_error", "with_prep_probability", float, "State preparation infidelity"
)
factory.add_mapping(
    "single_qubit_error", "with_p1_probability", float, "Single-qubit gate infidelity"
)
factory.add_mapping(
    "two_qubit_error", "with_p2_probability", float, "Two-qubit gate infidelity"
)
factory.add_mapping(
    "dark_count", "with_meas_0_probability", float, "Dark count probability"
)
factory.add_mapping(
    "detection_error", "with_meas_1_probability", float, "Bright state detection error"
)
factory.add_mapping(
    "motional_heating",
    "with_scale",
    lambda x: 1.0 + x * 0.01,  # Convert to scale factor
    "Motional heating rate",
)

# Typical ion trap parameters
config = {
    "state_prep_error": 0.001,
    "single_qubit_error": 0.0001,  # Very good single-qubit gates
    "two_qubit_error": 0.003,  # Main error source
    "dark_count": 0.001,  # Low dark count
    "detection_error": 0.005,  # Higher bright state error
    "motional_heating": 5.0,  # 5% heating effect
}

noise = factory.create_from_dict(config)
```

### Complex Noise Configuration

```python
config = {
    # Global settings
    "seed": 42,
    "scale": 1.2,  # Scale all errors by 20%
    # Make specific gates noiseless
    "noiseless_gates": ["H", "S", "T"],
    # State preparation
    "p_prep": 0.0005,
    # Single-qubit gates with Pauli distribution
    "p1_average": 0.001,
    "p1_pauli_model": {
        "X": 0.5,  # 50% X errors
        "Y": 0.3,  # 30% Y errors
        "Z": 0.2,  # 20% Z errors
    },
    # Two-qubit gates
    "p2_average": 0.008,
    "p2_pauli_model": {
        "IX": 0.25,
        "XI": 0.25,
        "XX": 0.5,
    },
    # Asymmetric measurement errors
    "p_meas_0": 0.002,  # 0->1 flip
    "p_meas_1": 0.005,  # 1->0 flip (higher)
}

noise = factory.create_from_dict(config)
```

### Factory with Defaults

Set factory-wide default values:

```python
factory = GeneralNoiseFactory()

# Set common defaults
factory.set_default("seed", 42)
factory.set_default("p1", 0.001)
factory.set_default("p2", 0.01)
factory.set_default("p_meas_0", 0.002)
factory.set_default("p_meas_1", 0.002)

# Empty config uses all defaults
noise1 = factory.create_from_dict({})

# Override specific values
noise2 = factory.create_from_dict(
    {
        "p2": 0.005,  # Override two-qubit error
        "scale": 0.5,  # Scale down all errors by 50%
    }
)
```

## Validation

Validate configurations before use:

```python
factory = GeneralNoiseFactory()

config = {
    "p1": "not_a_number",  # Type error
    "unknown_key": 123,  # Unknown key
    "p2": 0.01,  # Valid
}

errors = factory.validate_config(config)
print(errors)
# {
#     'p1': "could not convert string to float: 'not_a_number'",
#     'unknown_keys': "Unknown keys: {'unknown_key'}"
# }
```

## Best Practices

1. **Use descriptive keys**: When creating custom factories, use clear, descriptive parameter names that match your domain.

2. **Document your mappings**: Always provide descriptions when adding custom mappings.

3. **Validate early**: Use `validate_config()` before creating noise models in production code.

4. **Remove confusing aliases**: If the default aliases are confusing for your use case, remove them and keep only the primary keys.

5. **Version your configurations**: Store noise configurations in version-controlled JSON files for reproducibility.

6. **Use type converters**: Add appropriate converters (e.g., percentage to probability) to make configurations more intuitive.

## Integration with Existing Code

The factory integrates seamlessly with PECOS simulation APIs:

```python
from pecos.rslib import qasm_sim, GeneralNoiseFactory

# Create noise from configuration
factory = GeneralNoiseFactory()
noise = factory.create_from_dict(
    {
        "seed": 42,
        "p1": 0.001,
        "p2": 0.01,
    }
)

# Use with builder pattern
sim = qasm_sim(qasm).noise(noise).workers(4).build()
results = sim.run(1000)

# Or direct execution
results = qasm_sim(qasm).noise(noise).run(1000)
```

## Available Convenience Functions

```python
from pecos.rslib import (
    GeneralNoiseFactory,  # Main factory class
    create_noise_from_dict,  # Quick dict->noise conversion
    create_noise_from_json,  # Quick JSON->noise conversion
    IonTrapNoiseFactory,  # Pre-configured for ion traps
)
```
