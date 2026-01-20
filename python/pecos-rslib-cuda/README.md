# pecos-rslib-cuda

CUDA/cuQuantum Python bindings for PECOS quantum simulators.

This crate provides PyO3-based Python bindings for the Rust cuQuantum wrappers,
enabling GPU-accelerated quantum simulation from Python.

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- cuQuantum SDK installed

## Simulators

- `CuStateVec`: GPU-accelerated state vector simulation (~30 qubits)
- `CuStabilizer`: GPU-accelerated stabilizer simulation (1000s of qubits)

## Usage

```python
from pecos_rslib_cuda import CuStateVec, CuStabilizer, is_cuquantum_available

if is_cuquantum_available():
    # State vector simulation
    sim = CuStateVec(4)
    sim.h([0])
    sim.cx([0, 1])
    results = sim.mz([0, 1])

    # Stabilizer simulation (Clifford gates only)
    stab = CuStabilizer(100)
    stab.h([0])
    stab.cx([0, 1])
```
