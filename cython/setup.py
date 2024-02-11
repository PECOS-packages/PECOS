import contextlib
import shutil
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup

# Delete previous build folder
with contextlib.suppress(FileNotFoundError):
    current_location = Path(__file__).resolve().parent
    shutil.rmtree(current_location / "build")

compiler_flags = [
    "-std=c++17",
    "-O3",
    "-march=native",
    "-flto",
    "-fomit-frame-pointer",
    "-c",
]

ext_modules = [
    Extension(
        "cypecos.cysparsesim.cylib",
        sources=[
            "cypecos/cysparsesim/cysparsesim.pyx",
            "cypecos/cysparsesim/sparsesim.cpp",
        ],
        language="c++",
        extra_compile_args=compiler_flags,
        include_dirs=["./cypecos/cysparsesim/"],
    ),
]

for e in ext_modules:
    e.cython_directives = {
        "boundscheck": False,
        "wraparound": False,
    }

setup(
    ext_modules=cythonize(ext_modules, build_dir="build", language_level=3),
)
