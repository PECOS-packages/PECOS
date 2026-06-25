"""Compatibility helpers for cuQuantum import surface changes."""

try:
    from cuquantum.bindings import custatevec as cusv
except ImportError:
    from cuquantum import custatevec as cusv
