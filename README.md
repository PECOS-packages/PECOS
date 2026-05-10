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

Simulate a distance-3 repetition code with syndrome extraction using [Guppy](https://github.com/CQCL/guppylang), a pythonic quantum programming language:

```python
from pecos import Guppy, sim, state_vector, depolarizing_noise
from guppylang import guppy
from guppylang.std.quantum import qubit, cx, measure
from guppylang.std.builtins import array, result


@guppy
def repetition_code() -> None:
    # 3 data qubits encode logical |0⟩ = |000⟩
    d0, d1, d2 = qubit(), qubit(), qubit()

    # 2 ancillas for syndrome extraction
    s0, s1 = qubit(), qubit()

    # Measure parity between adjacent data qubits
    cx(d0, s0)
    cx(d1, s0)
    cx(d1, s1)
    cx(d2, s1)

    # Extract syndromes as an array
    result("syndrome", array(measure(s0), measure(s1)))

    # Measure data qubits (required by Guppy)
    _ = measure(d0), measure(d1), measure(d2)


# Run 10 shots with 10% depolarizing noise
noise = depolarizing_noise().with_uniform_probability(0.1)
results = sim(Guppy(repetition_code)).qubits(5).quantum(state_vector()).noise(noise).seed(42).run(10)
print(results["syndrome"])
# [[1, 1], [0, 1], [0, 0], [1, 1], [0, 0], [0, 1], [1, 1], [0, 0], [0, 1], [0, 1]]
```

Non-trivial syndromes like `[1, 0]`, `[0, 1]`, `[1, 1]` indicate detected errors that a decoder would use to identify and correct faults.

For OpenQASM, PHIR, or other formats, see the [documentation](#documentation). For a Rust example, see [For Rust Users](#for-rust-users) below.

## What Can You Do With PECOS?

- **Simulate quantum circuits** using fast simulators ideal for error correction research
- **Study quantum error correction codes** with tools for syndrome extraction, decoding, and analysis
- **Run hybrid quantum-classical programs** with support for classical control flow, conditionals, and Wasm
- **Add realistic noise** to understand how errors affect your circuits
- **Choose your backend**: stabilizer simulation, state vector, or GPU-accelerated options

## Documentation

For tutorials, API reference, and advanced features:

- [Getting Started Guide](docs/user-guide/getting-started.md) — Installation, first simulation, next steps
- [PECOS Concepts](docs/user-guide/pecos-concepts.md) — Detectors, observables, tracked operators, gates, and noise
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

Each crate also works standalone—use just `pecos-simulators` for simulation or `pecos-qec` for error correction without the full framework. Trait-based design makes it easy to swap implementations or integrate into your own tools.

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
