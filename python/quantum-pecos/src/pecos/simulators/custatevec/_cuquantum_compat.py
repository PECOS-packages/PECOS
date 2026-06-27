"""Optional CUDA-dependency availability for the Python cuStateVec backend.

This is the single place the cuStateVec simulator imports CuPy and the cuQuantum
bindings. It never raises at import time: when CuPy or a bindings-era cuQuantum
(``cuquantum.bindings``, introduced in cuQuantum 25.03) is missing, the names are
set to ``None`` and a classified, actionable reason is recorded. ``CuStateVec``
stays importable so ``import pecos`` works without CUDA; the failure surfaces
(loudly) only when the user constructs/uses ``CuStateVec``, via
``require_custatevec()``.
"""

from __future__ import annotations

__all__ = [
    "ComputeType",
    "cp",
    "cudaDataType",
    "custatevec_available",
    "custatevec_unavailable_reason",
    "cusv",
    "require_custatevec",
]

# Recommended install targets (kept in the messages so users get an actionable fix).
_CU13 = "cuquantum-python-cu13>=25.9.0"  # CUDA 13 (Turing/CC 7.5+)
_CU12 = "cuquantum-python-cu12>=25.3.0 (V100/Volta: >=25.3,<25.9)"  # CUDA 12

# These optional imports must NEVER break ``import pecos`` -- a broken/partial CUDA
# install can raise OSError/RuntimeError (not just ImportError) at import time, so we
# catch broadly and defer to a loud, actionable error at construction.
try:
    import cupy as cp
except Exception:  # noqa: BLE001 -- optional dependency; failures defer to construction
    cp = None

cusv = None
ComputeType = None
cudaDataType = None  # noqa: N816 -- mirrors cuQuantum's API name
_cuquantum_error: Exception | None = None
_cuquantum_version: str | None = None
try:
    from cuquantum import ComputeType, cudaDataType
    from cuquantum.bindings import custatevec as cusv
except Exception as exc:  # noqa: BLE001 -- optional dependency; failures defer to construction
    _cuquantum_error = exc
    try:
        import cuquantum

        _cuquantum_version = getattr(cuquantum, "__version__", "unknown")
    except ModuleNotFoundError:
        _cuquantum_version = None  # cuQuantum genuinely not installed
    except Exception:  # noqa: BLE001 -- installed but its import is broken
        # Present but broken: keep a non-None version so the reason surfaces the
        # original error (from _cuquantum_error) instead of claiming "not installed".
        _cuquantum_version = "unknown"


def _cuquantum_reason() -> str | None:
    """Classify why the bindings-era cuQuantum import failed (``None`` if it is fine)."""
    if cusv is not None and ComputeType is not None and cudaDataType is not None:
        return None
    if _cuquantum_version is None:
        return f"cuQuantum is not installed (install {_CU13} for CUDA 13 or {_CU12} for CUDA 12)"
    err = _cuquantum_error
    # Old (pre-25.03) cuQuantum: the bindings module/member is absent. Depending on the
    # layout this is either ModuleNotFoundError (no cuquantum.bindings package) or a plain
    # ImportError ("cannot import name 'custatevec' from 'cuquantum.bindings'", e.g. 24.11).
    pre_bindings = (
        isinstance(err, ModuleNotFoundError) and err.name in {"cuquantum.bindings", "cuquantum.bindings.custatevec"}
    ) or (isinstance(err, ImportError) and "cannot import name 'custatevec'" in str(err))
    if pre_bindings:
        return (
            f"cuQuantum {_cuquantum_version} predates the cuquantum.bindings API (requires >= 25.03); "
            f"upgrade to {_CU13} (CUDA 13) or {_CU12} (CUDA 12)"
        )
    # Bindings-era cuQuantum is present but importing custatevec failed for another reason
    # (e.g. a broken/partial install). Surface the original error rather than blaming version.
    return f"cuQuantum {_cuquantum_version} is installed but importing custatevec failed: {err}"


def _build_unavailable_reason() -> str | None:
    parts = []
    if cp is None:
        parts.append("CuPy is not installed (install cupy-cuda13x for CUDA 13 or cupy-cuda12x for CUDA 12)")
    cuquantum_reason = _cuquantum_reason()
    if cuquantum_reason is not None:
        parts.append(cuquantum_reason)
    if not parts:
        return None
    return "CuStateVec requires CuPy and a bindings-era cuQuantum. " + "; ".join(parts) + "."


_UNAVAILABLE_REASON = _build_unavailable_reason()


def custatevec_available() -> bool:
    """Whether the Python cuStateVec backend can be constructed in this environment."""
    return _UNAVAILABLE_REASON is None


def custatevec_unavailable_reason() -> str | None:
    """Human-readable reason CuStateVec is unavailable, or ``None`` if it is available."""
    return _UNAVAILABLE_REASON


def require_custatevec() -> None:
    """Raise an actionable error if the Python cuStateVec backend is unavailable."""
    if _UNAVAILABLE_REASON is not None:
        raise RuntimeError(_UNAVAILABLE_REASON)
