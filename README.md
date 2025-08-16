# ![PECOS](images/pecos_logo.svg)

[![PyPI version](https://badge.fury.io/py/quantum-pecos.svg)](https://badge.fury.io/py/quantum-pecos)
[![Documentation Status](https://readthedocs.org/projects/quantum-pecos/badge/?version=latest)](https://quantum-pecos.readthedocs.io/en/latest/?badge=latest)
[![Python versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://img.shields.io/badge/python-3.9%2C%203.10%2C%203.11-blue.svg)
[![Supported by Quantinuum](https://img.shields.io/badge/supported_by-Quantinuum-blue)](https://www.quantinuum.com/)

**Performance Estimator of Codes On Surfaces (PECOS)** is a library/framework dedicated to the study, development, and
evaluation of quantum error-correction protocols. It also offers tools for the study and evaluation of hybrid
quantum/classical compute execution models.

Initially conceived and developed in 2014 to verify lattice-surgery procedures presented in
[arXiv:1407.5103](https://arxiv.org/abs/1407.5103) and released publicly in 2018, PECOS filled the gap in
the QEC/QC tools available at that time. Over the years, it has grown into a framework for studying general QECCs and
hybrid computation.

## Features

- Quantum Error-Correction Tools: Advanced tools for studying quantum error-correction protocols and error models.
- Hybrid Quantum/Classical Execution: Evaluate advanced hybrid compute models, including support for classical compute,
calls to Wasm VMs, conditional branching, and more.
- Fast Simulation: Leverages a fast stabilizer simulation algorithm.
- Multi-language extensions: Core functionalities implemented via Rust for performance and safety. Additional add-ons
and extension support in C/C++ via Cython.
- QIR Support: Execute Quantum Intermediate Representation programs (requires LLVM version 14 with the 'llc' tool).

## Getting Started

Explore the capabilities of PECOS by delving into the [documentation](https://quantum-pecos.readthedocs.io).

## Repository Structure

PECOS now consists of multiple interconnected components:

- `/python/`: Contains Python packages
  - `/python/quantum-pecos/`: Main Python package (imports as `pecos`)
  - `/python/pecos-rslib/`: Python package with Rust extensions that utilize the `pecos` crate
- `/crates/`: Contains Rust crates
  - `/crates/pecos/`: Main Rust crate that collects the functionality of the other crates into one library
  - `/crates/pecos-core/`: Core Rust functionalities
  - `/crates/pecos-qsims/`: A collection of quantum simulators
  - `/crates/pecos-qec/`: Rust code for analyzing and exploring quantum error correction (QEC)
  - `/crates/pecos-qasm/`: Implementation of QASM parsing and execution
  - `/crates/pecos-qir/`: Implementation of QIR (Quantum Intermediate Representation) execution
  - `/crates/pecos-engines/`: Quantum and classical engines for simulations
  - `/crates/pecos-cli/`: Command-line interface for PECOS
  - `/crates/pecos-python/`: Rust code for Python extensions
  - `/crates/benchmarks/`: A collection of benchmarks to test the performance of the crates
- `/julia/`: Contains Julia packages (experimental)
  - `/julia/PECOS.jl/`: Main Julia package
  - `/julia/pecos-julia-ffi/`: Rust FFI library for Julia bindings

### Quantum Error Correction Decoders

PECOS includes LDPC (Low-Density Parity-Check) quantum error correction decoders as optional components. See [DECODERS.md](DECODERS.md) for detailed information about:
- LDPC decoder algorithms and variants
- How to build and use decoders
- Performance considerations
- Architecture and development guide

You may find most of these crates in crates.io if you wish to utilize only a part of PECOS, e.g., the simulators.

## Versioning

We follow semantic versioning principles. However, before version 1.0.0, the MAJOR.MINOR.BUG format sees the roles
of MAJOR and MINOR shifted down a step. This means potential breaking changes might occur between MINOR increments, such
as moving from versions 0.1.0 to 0.2.0.

All Python packages and all Rust crates will have the same version amongst their
respective languages; however, Python and Rust versioning will differ.

## Latest Development

Stay updated with the latest developments on the
[PECOS Development branch](https://quantum-pecos.readthedocs.io/en/development/).

## Installation

### Python Package

To install the main Python package for general usage:

```sh
pip install quantum-pecos
```

This will install both `quantum-pecos` and its dependency `pecos-rslib`.

For optional dependencies:

```sh
pip install quantum-pecos[all]
```

**NOTE:** The `quantum-pecos` package is imported like: `import pecos` and not `import quantum_pecos`.

**NOTE:** To install pre-releases (the latest development code) from pypi you may have to specify the version you are
interested like so (e.g., for version `0.6.0.dev5`):
```sh
pip install quantum-pecos==0.6.0.dev5
```

**NOTE:** Certain simulators have special requirements and are not installed by the command above. Installation instructions for
these are provided [here](#simulators-with-special-requirements).


### Rust Crates

To use PECOS in your Rust project, add the following to your `Cargo.toml`:

```toml
[dependencies]
pecos = "0.x.x"  # Replace with the latest version
```

#### Optional Dependencies

- **LLVM version 14**: Required for QIR (Quantum Intermediate Representation) support
  - Linux: `sudo apt install llvm-14`
  - macOS: `brew install llvm@14`
  - Windows: Download LLVM 14.x installer from [LLVM releases](https://releases.llvm.org/download.html#14.0.0)

  **Note**: Only LLVM version 14.x is compatible. LLVM 15 or later versions will not work with PECOS's QIR implementation.

  If LLVM 14 is not installed, PECOS will still function normally but QIR-related features will be disabled.

### Julia Package (Experimental)

PECOS also provides experimental Julia bindings. To use the Julia package from the development branch:

```julia
using Pkg
Pkg.add(url="https://github.com/PECOS-packages/PECOS#dev", subdir="julia/PECOS.jl")
```

Then you can use it:

```julia
using PECOS
println(pecos_version())  # Prints PECOS version
```

**Note**: The Julia package requires the Rust FFI library to be built. Currently, you need to build it locally:
1. Clone the repository
2. Build the FFI library: `cd julia/pecos-julia-ffi && cargo build --release`
3. Add the package locally: `Pkg.develop(path="julia/PECOS.jl")`

## Development Setup

If you are interested in editing or developing the code in this project, see this
[development documentation](docs/development/DEVELOPMENT.md) to get started.

## Simulators with special requirements

Certain simulators from `pecos.simulators` require external packages that are not installed by `pip install .[all]`.

- `CuStateVec` requires a Linux machine with an NVIDIA GPU (see requirements [here](https://docs.nvidia.com/cuda/cuquantum/latest/getting_started/getting_started.html#dependencies-custatevec-label)). PECOS' dependencies are
specified in the `[cuda]` section of `pyproject.toml`, however, installation via `pip` is not reliable. The recommended method of installation is via `conda`, as discussed [here](https://docs.nvidia.com/cuda/cuquantum/latest/getting_started/getting_started.html#installing-cuquantum). Note that there might be conflicts between `conda` and `venv`; if you intend to use `CuStateVec`, you may follow the installation instructions for PECOS within a `conda` environment without involving the `venv` commands.
- `MPS` uses `pytket-cutensornet` (see [repository](https://github.com/CQCL/pytket-cutensornet)) and can be installed via `pip install .[cuda]`. These
simulators use NVIDIA GPUs and cuQuantum. Unfortunately, installation of cuQuantum does not currently work via `pip`.
Please follow the instructions specified above for `CuStateVec` to install cuQuantum.

## Uninstall

To uninstall:

```sh
pip uninstall quantum-pecos
```

## Citing

For publications utilizing PECOS, kindly cite PECOS such as:

```bibtex
@misc{pecos,
 author={Ciar\'{a}n Ryan-Anderson},
 title={PECOS: Performance Estimator of Codes On Surfaces},
 publisher = {GitHub},
 journal = {GitHub repository},
 howpublished={\url{https://github.com/PECOS-packages/PECOS}},
 URL = {https://github.com/PECOS-packages/PECOS},
 year={2018}
}
```
And/or the PhD thesis PECOS was first described in:
```bibtex
@phdthesis{crathesis,
 author={Ciar\'{a}n Ryan-Anderson},
 school = {University of New Mexico},
 title={Quantum Algorithms, Architecture, and Error Correction},
 journal={arXiv:1812.04735},
 URL = {https://digitalrepository.unm.edu/phyc_etds/203},
 year={2018}
}
```

You can also use the [Zenodo DOI](https://zenodo.org/records/13700104), which would result in a bibtex like:
```bibtex
@software{pecos_[year],
  author       = {Ciar\'{a}n Ryan-Anderson},
  title        = {PECOS-packages/PECOS: [version]]},
  month        = [month],
  year         = [year],
  publisher    = {Zenodo},
  version      = {[version]]},
  doi          = {10.5281/zenodo.13700104},
  url          = {https://doi.org/10.5281/zenodo.13700104}
}
```


## License

This project is licensed under the Apache-2.0 License - see the [LICENSE](./LICENSE) and [NOTICE](NOTICE) files for
details.

## Supported by

[![Quantinuum](./images/Quantinuum_(word_trademark).svg)](https://www.quantinuum.com/)
