# ![PECOS](images/pecos_logo.svg)

[![PyPI version](https://badge.fury.io/py/quantum-pecos.svg)](https://badge.fury.io/py/quantum-pecos)
[![Crates.io](https://img.shields.io/crates/v/pecos.svg?color=brightgreen)](https://crates.io/crates/pecos)
[![Documentation Status](https://readthedocs.org/projects/quantum-pecos/badge/?version=latest)](https://quantum-pecos.readthedocs.io/en/latest/?badge=latest)
[![Python versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://img.shields.io/badge/python-3.10%2C%203.11%2C%203.12%2C%203.13%2C%203.14-blue.svg)
[![Supported by Quantinuum](https://img.shields.io/badge/supported_by-Quantinuum-blue)](https://www.quantinuum.com/)

[Installation](#installation) · [Quick Example](#quick-example) · [Documentation](#documentation) · [Rust](#for-rust-users) · [Citing](#citing)

**PECOS** (Performance Estimator of Codes On Surfaces) is a framework/library for exploring, developing, and evaluating quantum error correction protocols and hybrid quantum-classical programs.

Quantum error correcting since 2014. Fast simulators, from stabilizer to GPU. User-friendly Python API. Blazingly fast Rust core. Supported by Quantinuum.

## Installation

**Python:**
```bash
pip install quantum-pecos
```

**Rust:** Add to your `Cargo.toml`:
```toml
pecos = { version = "0.1", features = ["qasm"] }
```

For Julia or optional features (LLVM, CUDA), see the [Getting Started Guide](docs/user-guide/getting-started.md).

## Quick Example

Create and simulate a Bell state—an entangled pair of qubits:

```python
from pecos import sim, Qasm

# Define a Bell state circuit
circuit = Qasm(
    """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
"""
)

# Run 10 shots
results = sim(circuit).seed(42).run(10)
print(results.to_binary_dict())  # {"c": ["00", "11", "00", ...]} - qubits always match!
```

The results show `"00"` (both qubits measured `|0⟩`) and `"11"` (both measured `|1⟩`)—never `"01"` or `"10"`. That's quantum entanglement in action.

For a Rust example, see [For Rust Users](#for-rust-users) below.

## What Can You Do With PECOS?

- **Simulate quantum circuits** using fast simulators ideal for error correction research
- **Study quantum error correction codes** with tools for syndrome extraction, decoding, and analysis
- **Run hybrid quantum-classical programs** with support for classical control flow, conditionals, and Wasm
- **Add realistic noise** to understand how errors affect your circuits
- **Choose your backend**: stabilizer simulation, state vector, or GPU-accelerated options

## Documentation

For tutorials, API reference, and advanced features:

- [Getting Started Guide](docs/user-guide/getting-started.md) — Installation, first simulation, next steps
- [Simulators Guide](docs/user-guide/simulators.md) — Choosing the right backend
- [Noise Model Builders](docs/user-guide/noise-model-builders.md) — Adding realistic noise
- [Decoders Guide](docs/user-guide/decoders.md) — Quantum error correction decoding
- [Full Documentation](https://quantum-pecos.readthedocs.io) — Complete API reference

## Versioning

Before version 1.0.0, breaking changes may occur between minor versions (e.g., 0.1.0 → 0.2.0). We recommend pinning to a specific version in production.

## Citing

If you use PECOS in your research, please cite:

```bibtex
@misc{pecos,
 author={Ciar\'{a}n Ryan-Anderson},
 title={PECOS: Performance Estimator of Codes On Surfaces},
 howpublished={\url{https://github.com/PECOS-packages/PECOS}},
 year={2018}
}
```

For additional citation formats (PhD thesis, Zenodo DOI), see the [full documentation](https://quantum-pecos.readthedocs.io).

## License

Apache-2.0 — see [LICENSE](./LICENSE) for details.

---

## For Rust Users

The [`pecos`](https://crates.io/crates/pecos) crate is the main entry point—a metacrate that re-exports functionality from the underlying crates. Enable features for what you need:

```toml
[dependencies]
pecos = { version = "0.1", features = ["qasm", "phir"] }
```

Common features: `qasm` (OpenQASM support), `phir` (PHIR support), `llvm` (LLVM IR execution), `cli` (command-line tools). See [docs.rs/pecos](https://docs.rs/pecos) for the full list.

Each crate also works standalone—use just `pecos-qsim` for simulation or `pecos-qec` for error correction without the full framework. Trait-based design makes it easy to swap implementations or integrate into your own tools.

---

## For Contributors

### Repository Structure

- `/python/quantum-pecos/` — Main Python package (imports as `pecos`)
- `/python/pecos-rslib/` — Rust extensions for Python
- `/crates/pecos/` — Main Rust metacrate (re-exports other crates)
- `/crates/pecos-*/` — Individual Rust crates (simulators, engines, etc.)
- `/julia/` — Experimental Julia bindings
- `/docs/` — Documentation source

Both Rust and Python are designed to be modular. Extend or replace components without forking.

See the [Development Guide](docs/development/DEVELOPMENT.md) to get started contributing.

---

[![Quantinuum](./images/Quantinuum_(word_trademark).svg)](https://www.quantinuum.com/)
