# ![PECOS](branding/logo/pecos_logo_v2.png)

[![PyPI version](https://badge.fury.io/py/quantum-pecos.svg)](https://badge.fury.io/py/quantum-pecos)
[![Documentation Status](https://readthedocs.org/projects/quantum-pecos/badge/?version=latest)](https://quantum-pecos.readthedocs.io/en/latest/?badge=latest)
[![Python versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://img.shields.io/badge/python-3.9%2C%203.10%2C%203.11-blue.svg)
[![Supported by Quantinuum](https://img.shields.io/badge/supported_by-Quantinuum-blue)](https://www.quantinuum.com/)

**Performance Estimator of Codes On Surfaces (PECOS)** is a library/framework dedicated to the study, development, and
evaluation of quantum error-correction protocols. It offers tools for the study and evaluation of hybrid
quantum/classical compute execution models for NISQ algorithms and beyond.

Initially conceived and developed in 2014 to verify lattice-surgery procedures presented in
[arXiv:1407.5103](https://arxiv.org/abs/1407.5103) and released publicly in 2018, PECOS filled a significant gap in
the QEC/QC tools available at that time. Over the years, it has grown into a framework for studying general QECCs and
hybrid computation.

With an emphasis on clarity, flexibility, and performance and catering to both QEC students and developers, PECOS is
refined continually with these attributes in mind.

## Features

- Quantum Error-Correction Tools: Advanced tools for studying quantum error-correction protocols and error models.
- Hybrid Quantum/Classical Execution: Evaluate advanced hybrid compute models, including support for classical compute,
calls to Wasm VMs, conditional branching, and more.
- Fast Simulation: Leverage the fast stabilizer-simulation algorithm.
- Extensible: Add-ons and extensions support in C and C++ via Cython.

## Getting Started

Explore the capabilities of PECOS by delving into the [official documentation](https://quantum-pecos.readthedocs.io).

## Versioning

We follow semantic versioning principles. However, before version 1.0.0, the MAJOR.MINOR.BUG format sees the roles
of MAJOR and MINOR shifted down a step. This means potential breaking changes might occur between MINOR increments, such
as moving from versions 0.1.0 to 0.2.0.

## Latest Development

Stay updated with the latest developments on the
[PECOS Development branch](https://quantum-pecos.readthedocs.io/en/development/).

## Installation

1. Clone or download the desired version of PECOS.
2. Navigate to the root directory, where `pyproject.toml` is located.
3. Install using pip:

```sh
pip install .
```

To install optional dependencies, such as for Wasm support or state vector simulations, see `pyproject.toml` for list of
options. To install all optional dependencies use:

```sh
pip install .[all]
```

Certain simulators have special requirements and are not installed by the command above. Installation instructions for these are provided [here](#simulators-with-special-requirements).

For development, use (while including installation options as necessary):

On Linux/Mac:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools
pip install -r requirements.txt
make metadeps
pre-commit install
pip install -e .
```

On Windows:

```sh
python -m venv .venv
.\venv\Scripts\activate
pip install -U pip setuptools
pip install -r requirements.txt
make metadeps
pre-commit install
pip install -e .
```

See `Makefile` for other useful commands.

Tests can be run using:

```sh
pytest tests
```

### Simulators with special requirements

Certain simulators from `pecos.simulators` require external packages that are not installed by `pip install .[all]`.

- `QuEST` is installed along with the python package `pyquest` when calling `pip install .[all]`. However, it uses 64-bit float point precision by default, and if you wish to make use of 32-bit float point precision you will need to follow the installation instructions provided by the developers [here](https://github.com/rrmeister/pyQuEST/tree/develop).
- `CuStateVec` requires a Linux machine with an NVIDIA GPU (see requirements [here](https://docs.nvidia.com/cuda/cuquantum/latest/getting_started/getting_started.html#dependencies-custatevec-label)). PECOS' dependencies are specified in the `[cuda]` section of `pyproject.toml`, however, installation via `pip` is not reliable. The recommended method of installation is via `conda`, as discussed [here](https://docs.nvidia.com/cuda/cuquantum/latest/getting_started/getting_started.html#installing-cuquantum). Note that there might be conflicts between `conda` and `venv`; if you intend to use `CuStateVec`, you may follow the installation instructions for PECOS within a `conda` environment without involving the `venv` commands.
- `MPS` uses `pytket-cutensornet` (see [repository](https://github.com/CQCL/pytket-cutensornet)) and can be installed via `pip install .[cuda]`. This simulators uses NVIDIA GPUs and cuQuantum. Unfortunately, installation of cuQuantum does not currently work via `pip`. Please follow the instructions specified above for `CuStateVec` to install cuQuantum.

## Uninstall

To uninstall:

```sh
pip uninstall quantum-pecos
```

## Citing

For publications utilizing PECOS, kindly cite PECOS such as:

```bibtex
@misc{pecos,
 author={Ciaran Ryan-Anderson},
 title={PECOS: Performance Estimator of Codes On Surfaces},
 journal={GitHub},
 howpublished={\url{https://github.com/PECOS-packages/PECOS}},
 URL = {https://github.com/PECOS-packages/PECOS},
 year={2018}
}

@phdthesis{crathesis,
 author={Ciaran Ryan-Anderson},
 school = {University of New Mexico},
 title={Quantum Algorithms, Architecture, and Error Correction},
 journal={arXiv:1812.04735},
 year={2018}
}
```

## License

This project is licensed under the Apache-2.0 License - see the [LICENSE](./LICENSE) and [NOTICE](NOTICE) files for
details.

## Supported by

[![Quantinuum](./images/Quantinuum_(word_trademark).svg)](https://www.quantinuum.com/)
