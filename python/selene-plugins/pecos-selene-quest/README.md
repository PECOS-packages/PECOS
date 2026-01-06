# PECOS Quest Selene Plugin

A [Selene](https://github.com/Quantinuum/selene) quantum emulator plugin providing access to the [QuEST](https://github.com/quest-kit/QuEST) (Quantum Exact Simulation Toolkit) simulator through the PECOS wrapper.

## About QuEST

This plugin wraps **QuEST** (Quantum Exact Simulation Toolkit), a high-performance quantum simulator developed by the [QuEST-Kit team](https://github.com/quest-kit). QuEST is licensed under the MIT License.

**QuEST Repository:** https://github.com/quest-kit/QuEST

QuEST supports:
- State vector simulation
- Density matrix simulation
- GPU acceleration (CUDA)
- Arbitrary rotation angles (non-Clifford gates)

If you use QuEST in your research, please cite the following papers:

- Jones, T., Brown, A., Bush, I. & Benjamin, S.C. *QuEST and High Performance Simulation of Quantum Computers.* Sci Rep 9, 10736 (2019). https://doi.org/10.1038/s41598-019-47174-9
- Jones, T. & Sherbert, K. *QuESTlink and QASMlink—Mathematica packages for high-performance simulation of quantum computers.* (2022). https://arxiv.org/abs/2210.16724

## Overview

This plugin provides QuEST simulator backends for Selene, using the PECOS QuEST wrapper. It supports both state vector and density matrix simulation modes, with optional GPU acceleration.

Memory requirements:
- State vector: 16 bytes * 2^n_qubits
- Density matrix: 16 bytes * 4^n_qubits

## Installation

```bash
pip install pecos-selene-quest
```

## Usage

```python
from selene_sim.build import build
from pecos_selene_quest import QuestPlugin, SimulatorMode

# Default: CPU state vector simulation
simulator = QuestPlugin()

# Density matrix simulation
simulator = QuestPlugin(mode=SimulatorMode.DENSITY_MATRIX)

# GPU-accelerated state vector simulation
simulator = QuestPlugin(use_gpu=True)

# GPU-accelerated density matrix simulation
simulator = QuestPlugin(mode=SimulatorMode.DENSITY_MATRIX, use_gpu=True)

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

- `mode` (SimulatorMode): Simulation mode - `STATE_VECTOR` (default) or `DENSITY_MATRIX`
- `use_gpu` (bool): Enable GPU acceleration (default: False). Requires CUDA.
- `random_seed` (int, optional): Seed for the random number generator for deterministic results.

## GPU Support

GPU acceleration requires:
- CUDA toolkit installed
- A compatible NVIDIA GPU
- The plugin built with GPU support (automatically detected during build)

If GPU is requested but not available, a clear error message will be shown.

## Building from Source

This package requires Rust and the QuEST C library to build. The Rust components will be automatically compiled during installation.

```bash
# From the PECOS repository root
cd python/selene-plugins/pecos-selene-quest
pip install -e ".[test]"
```

## Running Tests

```bash
pytest tests/

# Force GPU tests to run (will fail if GPU unavailable)
PECOS_TEST_GPU=1 pytest tests/
```

## License

This PECOS plugin is licensed under Apache-2.0.

The underlying QuEST library is licensed under the MIT License. See the [QuEST repository](https://github.com/quest-kit/QuEST) for details.
