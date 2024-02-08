![PECOS](docs/images/pecos_large_logo.png)
=======================================

[![PyPI version](https://badge.fury.io/py/quantum-pecos.svg)](https://badge.fury.io/py/quantum-pecos)
[![Documentation Status](https://readthedocs.org/projects/quantum-pecos/badge/?version=latest)](https://quantum-pecos.readthedocs.io/en/latest/?badge=latest)
[![Python Status](https://img.shields.io/badge/python-3.5.2%2C%203.6%2C%203.7-brightgreen.svg)](https://img.shields.io/badge/python-3.5.2%2C%203.6%2C%203.7-brightgreen.svg)

PECOS (Performance Estimator of Codes On Surfaces) is a Python framework for studying, developing, and evaluating 
quantum error-correction protocols.

- Author: Ciarán Ryan-Anderson
- Language: Python 3.5.2+ (with optional C and C++ extensions)

## Contact

 For questions or suggestions, please feel free to contact the author:
 
   - Ciarán Ryan-Anderson, ciaran.ryan-anderson@quantinuum.com
   
## Getting Started

To get started, check out the documentation in the "docs" folder or find it online at:

https://quantum-pecos.readthedocs.io
   
## Latest Development

See the following branch for the latest version of PECOS under development:

https://github.com/PECOS-packages/PECOS/tree/development

BEAWARE: There are some changes planned in 0.2.dev that may break some backwards compatibility with 0.1. Although, we try to minimize breaks to backwards compatibility.

## Requirements
- Python 3.5.2+
- NumPy 1.15+
- SciPy 1.1+
- Matplotlib 2.2+
- NetworkX 2.1+

## Optional Dependencies

- Cython (to compile optional C/C++ extensions)
- pytest 3.0+ (to run tests)
- Sphinx 2.7.6+ (to compile documentation)

## License

PECOS is licensed under the [Apache 2.0 license](https://github.com/PECOS-packages/PECOS/blob/master/LICENSE)

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
