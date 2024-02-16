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
