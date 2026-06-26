"""Compatibility helpers for cuQuantum import surface changes."""

_BINDINGS_MISSING_NAMES = {"cuquantum.bindings", "cuquantum.bindings.custatevec"}
_BINDINGS_MISSING_MEMBER = "cannot import name 'custatevec' from 'cuquantum.bindings'"


def _should_fallback_to_legacy(exc: ImportError) -> bool:
    if isinstance(exc, ModuleNotFoundError):
        return exc.name in _BINDINGS_MISSING_NAMES
    return _BINDINGS_MISSING_MEMBER in str(exc)


try:
    from cuquantum.bindings import custatevec as cusv
except ImportError as exc:
    if not _should_fallback_to_legacy(exc):
        raise
    from cuquantum import custatevec as cusv
