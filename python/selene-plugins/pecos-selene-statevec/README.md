# PECOS StateVec Selene Plugin

A state vector simulator plugin for the [Selene](https://github.com/CQCL/selene) quantum emulator using the PECOS state vector implementation.

## Overview

This plugin provides a full state vector simulator backend for Selene. Unlike stabilizer simulators, it can simulate arbitrary rotation angles, making it suitable for any quantum circuit.

The memory requirement scales exponentially with the number of qubits (16 bytes * 2^n_qubits).

## Installation

```bash
pip install pecos-selene-statevec
```

## Usage

```python
from selene_sim.build import build
from pecos_selene_statevec import StateVecPlugin

# Create a plugin instance
simulator = StateVecPlugin()

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
cd python/pecos-selene-statevec
pip install -e ".[test]"
```

## Running Tests

```bash
pytest tests/
```

## License

Apache-2.0
