![PECOS](docs/images/pecos_large_logo.png)
=======================================

[![PyPI version](https://badge.fury.io/py/quantum-pecos.svg)](https://badge.fury.io/py/quantum-pecos)
[![Documentation Status](https://readthedocs.org/projects/quantum-pecos/badge/?version=latest)](https://quantum-pecos.readthedocs.io/en/latest/?badge=latest)

PECOS (Performance Estimator of Codes On Surfaces) is a Python framework for studying, developing, and evaluating 
quantum error-correction protocols.

- Author: Ciarán Ryan-Anderson
- Language: Python 3.5.2+ (with optional C and C++ extensions)

## Contact
   - Ciarán Ryan-Anderson, ciaran@pecos.io

## Getting Started

To get started, check out the documentation in the "docs" folder. You can also see the documentation on Read the Docs 
here:

https://quantum-pecos.readthedocs.io

## Requirements
- Python 3.5.2+
- NumPy 1.15+
- SciPy 1.1+
- Matplotlib 2.2+
- NetworkX 2.1+

## Optional Dependencies

- Cython (to compile optional C/C++ extensions)
- pytest 3.0+ (to run tests)
- Sphinx 2.7.6+ (to compile the documentation)
- ProjectQ (to take advantage of the full quantum-state simulators made available via ProjectQ)
- Cirq (to take advantage of the full quantum-state simulators made available via Cirq) [WARNING: In early development.] 
## Installation

To install using pip run the command:
```
pip install quantum-pecos
```

To install from GitHub go to:

https://github.com/PECOS-packages/PECOS

Then, download/unzip or clone the version of PECOS you would like to use. Next, navigate to the root of the package 
(where setup.py is located) and run:
```
pip install .
```

To install and continue to develop the version of PECOS located in the install folder, run the
 following instead:
```
pip install -e .
```

## Uninstall

To uninstall run:
```
pip uninstall quantum-pecos
```