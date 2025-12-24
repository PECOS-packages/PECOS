# PECOS Qulacs Selene Plugin

A [Selene](https://github.com/CQCL/selene) quantum emulator plugin providing access to the [Qulacs](https://github.com/qulacs/qulacs) simulator through the PECOS wrapper.

## About Qulacs

This plugin wraps **Qulacs**, a high-performance quantum circuit simulator developed by the [Qulacs team](https://github.com/qulacs). Qulacs is licensed under the MIT License.

**Qulacs Repository:** https://github.com/qulacs/qulacs

Qulacs supports:
- State vector simulation
- Arbitrary rotation angles (non-Clifford gates)
- High-performance CPU execution with SIMD optimization

If you use Qulacs in your research, please cite the following paper:

- Suzuki, Y., Kawase, Y., Masumura, Y. et al. *Qulacs: a fast and versatile quantum circuit simulator for research purpose.* Quantum 5, 559 (2021). https://arxiv.org/abs/2011.13524

## Overview

This plugin provides a Qulacs state vector simulator backend for Selene, using the PECOS Qulacs wrapper. Currently only state vector simulation is supported.

Memory requirements:
- State vector: 16 bytes * 2^n_qubits

## Installation

```bash
pip install pecos-selene-qulacs
```

## Usage

```python
from selene_sim.build import build
from pecos_selene_qulacs import QulacsPlugin

# Create a plugin instance
simulator = QulacsPlugin()

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

- `mode` (SimulatorMode): Simulation mode - currently only `STATE_VECTOR` is supported.
- `random_seed` (int, optional): Seed for the random number generator for deterministic results.

## Building from Source

This package requires Rust and the Qulacs C++ library to build. The Rust components will be automatically compiled during installation.

```bash
# From the PECOS repository root
cd python/selene-plugins/pecos-selene-qulacs
pip install -e ".[test]"
```

## Running Tests

```bash
pytest tests/
```

## License

This PECOS plugin is licensed under Apache-2.0.

The underlying Qulacs library is licensed under the MIT License. See the [Qulacs repository](https://github.com/qulacs/qulacs) for details.
