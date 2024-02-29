# Copyright 2024 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

# Ruff is acting up with this file and acting differently between OSes
# ruff: noqa
from Cython.Build import cythonize
from setuptools import Extension, setup

compiler_flags = [
    "-std=c++17",
    "-O3",
    "-march=native",
    "-flto",
    "-fomit-frame-pointer",
    "-c",
]

# TODO: Add MSVC specific flags

ext_modules = [
    Extension(
        "cypecos.cysparsesim.cysparsesim",
        sources=[
            "cypecos/cysparsesim/cysparsesim.pyx",
            "../cpp/sparsesim/sparsesim.cpp",
        ],
        include_dirs=["../cpp/sparsesim"],
        language="c++",
        extra_compile_args=compiler_flags,
    ),
]

for e in ext_modules:
    e.cython_directives = {
        "boundscheck": False,
        "wraparound": False,
    }

setup(
    ext_modules=cythonize(
        ext_modules,
        build_dir="build",
        language_level=3,
    ),
)