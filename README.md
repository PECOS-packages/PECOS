# ![PECOS](branding/logo/pecos_logo_v2.svg)

[![PyPI version](https://badge.fury.io/py/quantum-pecos.svg)](https://badge.fury.io/py/quantum-pecos)
[![Documentation Status](https://readthedocs.org/projects/quantum-pecos/badge/?version=latest)](https://quantum-pecos.readthedocs.io/en/latest/?badge=latest)
[![Python versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://img.shields.io/badge/python-3.9%2C%203.10%2C%203.11-blue.svg)
[![Supported by Quantinuum](https://img.shields.io/badge/supported_by-Quantinuum-blue)](https://www.quantinuum.com/)

**Performance Estimator of Codes On Surfaces (PECOS)** is a library/framework dedicated to the study, development, and
evaluation of quantum error-correction protocols. It also offers tools for the study and evaluation of hybrid
quantum/classical compute execution models for NISQ algorithms and beyond.

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
  - `/crates/pecos-python/`: Rust code for Python extensions
  - `/crates/benchmarks/`: A collection of benchmarks to test the performance of the crates

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

## Development Setup

If you are interested in editing or developing the code in this project, see this
[development documentation](development.md) to get started.

## Simulators with special requirements

Certain simulators from `pecos.simulators` require external packages that are not installed by `pip install .[all]`.

- `QuEST` is installed along with the python package `pyquest` when calling `pip install .[all]`. However, it uses
64-bit float point precision by default, and if you wish to make use of 32-bit float point precision you will need to
follow the installation instructions provided by the developers [here](https://github.com/rrmeister/pyQuEST/tree/develop).
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
