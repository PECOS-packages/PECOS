from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import os
import shutil

# Delete previous build folder
current_location = os.path.dirname(os.path.abspath(__file__))
try:
    shutil.rmtree(os.path.join(current_location, 'build'))
except FileNotFoundError:
    pass

compiler_flags = ["-std=c++17", "-O3", "-march=native", "-flto",
"-fomit-frame-pointer", "-c", ]

ext_modules = [
    Extension('cypecos.cysparsesim.cylib',
              sources=[
                  "cypecos/cysparsesim/cysparsesim.pyx",
                  "cypecos/cysparsesim/sparsesim.cpp",
              ],
              language='c++',
              extra_compile_args=compiler_flags,
              include_dirs=['./cypecos/cysparsesim/'],
              ),
]

for e in ext_modules:
    e.cython_directives = {
        'boundscheck': False,
        'wraparound': False,
    }

setup(
    ext_modules=cythonize(ext_modules, build_dir="build", language_level=3),
)
