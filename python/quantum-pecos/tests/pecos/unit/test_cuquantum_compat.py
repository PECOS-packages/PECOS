# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

"""Tests for cuStateVec optional-dependency availability (``_cuquantum_compat``).

The module must never raise at import time, and must report a classified,
actionable reason (and raise it from ``require_custatevec``) when CuPy or a
bindings-era cuQuantum (>= 25.03) is missing -- never a silent ``None`` or a
legacy fallback. These tests mock the optional deps so they hold regardless of
whether real CuPy/cuQuantum is installed in the test environment.
"""

from __future__ import annotations

import builtins
import importlib.util
import sys
import types
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[3] / "src/pecos/simulators/custatevec/_cuquantum_compat.py"


def _load_module():
    """Execute a fresh copy of the compat module under the current mocks."""
    spec = importlib.util.spec_from_file_location("_test_cuquantum_compat", _MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop("_test_cuquantum_compat", None)


def _fake_cuquantum(monkeypatch, version: str, *, with_custatevec: bool) -> None:
    """Install a fake ``cuquantum`` package. Omitting ``custatevec`` reproduces the
    pre-25.03 shape where ``cuquantum.bindings`` exists but lacks the member."""
    cuquantum = types.ModuleType("cuquantum")
    cuquantum.__path__ = []
    cuquantum.__version__ = version
    cuquantum.ComputeType = object()
    cuquantum.cudaDataType = object()
    bindings = types.ModuleType("cuquantum.bindings")
    bindings.__path__ = []
    monkeypatch.setitem(sys.modules, "cuquantum", cuquantum)
    monkeypatch.setitem(sys.modules, "cuquantum.bindings", bindings)
    if with_custatevec:
        custatevec = types.ModuleType("cuquantum.bindings.custatevec")
        bindings.custatevec = custatevec
        monkeypatch.setitem(sys.modules, "cuquantum.bindings.custatevec", custatevec)
    else:
        # Evict any real submodule so `from cuquantum.bindings import custatevec` actually
        # fails (the test env may have a modern cuQuantum installed).
        monkeypatch.delitem(sys.modules, "cuquantum.bindings.custatevec", raising=False)


def _block_imports(monkeypatch, *top_levels: str) -> None:
    """Make the named top-level packages un-importable (overrides a real install)."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.split(".")[0] in top_levels:
            msg = f"No module named {name!r}"
            raise ModuleNotFoundError(msg, name=name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    for mod in list(sys.modules):
        if mod.split(".")[0] in top_levels:
            monkeypatch.delitem(sys.modules, mod, raising=False)


def test_available_when_cupy_and_bindings_present(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "cupy", types.ModuleType("cupy"))
    _fake_cuquantum(monkeypatch, "26.3.2", with_custatevec=True)

    compat = _load_module()

    assert compat.custatevec_available()
    assert compat.custatevec_unavailable_reason() is None
    assert compat.cusv is sys.modules["cuquantum.bindings.custatevec"]
    compat.require_custatevec()  # must not raise


def test_loud_when_cuquantum_too_old(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "cupy", types.ModuleType("cupy"))
    _fake_cuquantum(monkeypatch, "24.11.0", with_custatevec=False)

    compat = _load_module()

    assert not compat.custatevec_available()
    reason = compat.custatevec_unavailable_reason()
    assert "25.03" in reason
    assert "24.11.0" in reason
    with pytest.raises(RuntimeError, match=r"25\.03"):
        compat.require_custatevec()


def test_loud_when_cuquantum_missing(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "cupy", types.ModuleType("cupy"))
    _block_imports(monkeypatch, "cuquantum")

    compat = _load_module()

    assert not compat.custatevec_available()
    assert "not installed" in compat.custatevec_unavailable_reason()
    with pytest.raises(RuntimeError):
        compat.require_custatevec()


def test_loud_when_cupy_missing(monkeypatch) -> None:
    _block_imports(monkeypatch, "cupy")
    _fake_cuquantum(monkeypatch, "26.3.2", with_custatevec=True)

    compat = _load_module()

    assert not compat.custatevec_available()
    assert "CuPy" in compat.custatevec_unavailable_reason()
    with pytest.raises(RuntimeError, match="CuPy"):
        compat.require_custatevec()
