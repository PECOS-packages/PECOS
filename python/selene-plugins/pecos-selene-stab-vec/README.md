# PECOS StabVec Selene Plugin

A StabVec (Clifford+RZ) simulator plugin for the [Selene](https://github.com/Quantinuum/selene) quantum emulator using the PECOS sum-over-Cliffords implementation.

## Overview

This plugin provides a StabVec (Clifford+RZ) simulator backend for Selene. It handles Clifford gates efficiently and supports arbitrary RZ rotations via a sum-over-Cliffords decomposition.

The cost is polynomial in qubits and Clifford gates, but exponential in the number of non-Clifford (RZ) gates applied. This makes it well-suited for circuits with many qubits but few non-Clifford gates.

## Installation

```bash
pip install pecos-selene-stab-vec
```

## Usage

```python
from selene_sim.build import build
from pecos_selene_stab_vec import StabVecPlugin

# Create a plugin instance
simulator = StabVecPlugin()

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
cd python/selene-plugins/pecos-selene-stab-vec
pip install -e ".[test]"
```

## Running Tests

```bash
pytest tests/
```

## License

Apache-2.0
