![PECOS](docs/images/pecos_large_logo.png)
=======================================

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

## Installation

Download or clone this package and navigate to the root. Then:

To install to develop PECOS run:
```
pip install -e .
```

Otherwise run:
```
pip install setup.py
```

## Uninstall

To uninstall run:
```
pip uninstall quantum-pecos
```