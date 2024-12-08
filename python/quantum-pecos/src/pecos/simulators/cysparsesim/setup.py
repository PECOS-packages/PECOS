# Copyright 2019 The PECOS Developers
# Copyright 2018 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract
# DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Setup
=====.

The setup file for the Cython wrapped C++ version of SparseSim.

Notes:
-----
    Use the following to compile from command line:
    python setup.py build_ext --inplace

"""

import contextlib
import shutil
from distutils.core import setup
from distutils.extension import Extension
from pathlib import Path

from Cython.Build import cythonize

# Delete previous build folder
current_location = Path.parent(Path.resolve(__file__))
with contextlib.suppress(FileNotFoundError):
    shutil.rmtree(Path(current_location / "build"))

# compiler_flags = ["-std=c++11", "-Wall", "-fPIC", "-O2", "-O3", "-c", ]
compiler_flags = ["-std=c++11", "-W3", "-fPIC", "-O2", "-O3", "-c"]

ext_modules = [
    Extension(
        "cysparsesim",
        sources=[
            "src/cysparsesim.pyx",
            "src/sparsesim.cpp",
            "src/logical_sign.py",
        ],
        language="c++",
        extra_compile_args=compiler_flags,
        include_dirs=["./src"],
        # include_dirs=[np.get_include()],
    ),
]


for e in ext_modules:
    e.cython_directives = {
        "boundscheck": False,
        "wraparound": False,
    }


setup(
    name="state",
    ext_modules=cythonize(ext_modules, build_dir="build", language_level=3),
    script_args=["build_ext"],
    options={
        "build_ext": {"inplace": True},
    },
)
