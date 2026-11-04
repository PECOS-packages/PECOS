# PECOS StabActive Selene Plugin

A StabActive (Clifford+RZ) simulator plugin for the [Selene](https://github.com/Quantinuum/selene) quantum emulator using the PECOS active stabilizer-coordinate implementation.

## Overview

This plugin provides a StabActive (Clifford+RZ) simulator backend for Selene. It handles Clifford gates efficiently and supports arbitrary RZ rotations via dense active coordinates.

The cost is polynomial in qubits and Clifford gates, but exponential in the active width. This makes it well-suited for circuits with many qubits but few non-Clifford gates.

The plugin uses the platform maximum active width; available memory limits simulation.
Metrics `active_width` and `peak_active_width` report the current width and the peak
since shot start. State dumping is unsupported.

## Installation

```bash
pip install pecos-selene-stab-active
```

## Usage

```python
from selene_sim.build import build
from pecos_selene_stab_active import StabActivePlugin

# Create a plugin instance
simulator = StabActivePlugin()

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

- `random_seed` (int, optional): Seed for the random number generator for deterministic results.

## Building from Source

This package requires Rust to build. The Rust components will be automatically compiled during installation.

```bash
# From the PECOS repository root
cd python/selene-plugins/pecos-selene-stab-active
pip install -e ".[test]"
```

## Running Tests

```bash
pytest tests/
```

## License

Apache-2.0
