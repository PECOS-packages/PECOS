"""SLR test package.

The `__init__.py` here is load-bearing: without it, pytest's importlib
mode resolves `slr_tests/guppy/test_hugr_compilation.py` and `guppy/
test_hugr_compilation.py` to the same module name (`guppy.test_hugr_
compilation`), and the second-loaded file silently aliases to the first.
"""
