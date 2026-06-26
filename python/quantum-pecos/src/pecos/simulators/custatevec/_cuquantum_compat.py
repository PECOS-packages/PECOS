"""Compatibility helpers for cuQuantum import surface changes."""

try:
    from cuquantum.bindings import custatevec as cusv
except ModuleNotFoundError as exc:
    if exc.name not in {"cuquantum.bindings", "cuquantum.bindings.custatevec"}:
        raise
    from cuquantum import custatevec as cusv
