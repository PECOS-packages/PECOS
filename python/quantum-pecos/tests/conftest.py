"""Shared pytest setup for the Python test suite."""

# The test tree contains ``tests/pecos``. Pytest can import tests from that
# directory as a namespace package named ``pecos`` when an individual file is run
# in isolation. Import the installed/source package first so later
# ``import pecos`` statements resolve to the public PECOS package.
import pecos
