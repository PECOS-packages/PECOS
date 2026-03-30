# PECOS Stabilizer Selene Plugin

A stabilizer simulator plugin for the [Selene](https://github.com/Quantinuum/selene) quantum emulator using the PECOS stabilizer implementation.

## Overview

This plugin provides a Clifford simulator backend for Selene. As a stabilizer simulator, it can only simulate Clifford operations (rotations that are multiples of pi/2).

## Installation

```bash
pip install pecos-selene-stabilizer
```

## Usage

```python
from selene_sim.build import build
from pecos_selene_stabilizer import StabPlugin

# Create a plugin instance
simulator = StabPlugin()

# Or customize the angle threshold for Clifford approximation
simulator = StabPlugin(angle_threshold=1e-4)

# Use with Selene
runner = build(program)
results = list(
    runner.run_shots(
        simulator=simulator,
        n_qubits=10,
        n_shots=1000,
    )
)
```

## Parameters

- `angle_threshold` (float, default=1e-4): Angles within this threshold of a multiple of pi/2 will be rounded to that Clifford rotation. Must be greater than zero.
- `random_seed` (int, optional): Seed for the random number generator for deterministic results.

## Building from Source

This package requires Rust to build. The Rust components will be automatically compiled during installation.

```bash
# From the PECOS repository root
cd python/pecos-selene-stabilizer
pip install -e ".[test]"
```

## Running Tests

```bash
pytest tests/
```

## License

Apache-2.0
