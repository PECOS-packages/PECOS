#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  =========================================================================  #
#   Copyright 2018 National Technology & Engineering Solutions of Sandia,
#   LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS,
#   the U.S. Government retains certain rights in this software.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#  =========================================================================  #

"""
Setup script for PECOS.

For development install, use the following at the package root:

pip install -e .

"""

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

exec(open('pecos/version.py').read())  # Get __version__

setup(
    name='quantum-pecos',
    version=__version__,
    url='https://github.com/PECOS-packages/PECOS',
    author='Ciarán Ryan-Anderson',
    author_email='ciaran@pecos.io',
    description='PECOS (Performance Estimator of Codes On Surfaces) is a package designed to facilitate the evaluation '
                'and study of quantum error correcting codes.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires='>=3.5.2',
    install_requires=[
        'numpy>=1.15.0',
        'scipy>=1.1.0',
        'matplotlib>=2.2.0',
        'networkx>=2.1.0',
        'sortedcontainers>=2.0.0',
    ],
    tests_require=['pytest>=3.0.0'],
    extras_require={
        'all': ['cirq', 'cython', 'projectq'],
        'simulators': ['cirq', 'cython', 'projectq'],
    },
    license='Apache 2',
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    )
