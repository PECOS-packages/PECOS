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

"""Tests for cuQuantum import compatibility shims."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[3] / "src/pecos/simulators/custatevec/_cuquantum_compat.py"


def _load_module(module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, _MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(module_name, None)


def test_prefers_bindings_custatevec(monkeypatch) -> None:
    legacy = object()
    bindings = object()

    cuquantum = types.ModuleType("cuquantum")
    cuquantum.__path__ = []
    cuquantum.custatevec = legacy
    cuquantum_bindings = types.ModuleType("cuquantum.bindings")
    cuquantum_bindings.__path__ = []
    cuquantum_bindings.custatevec = bindings

    monkeypatch.setitem(sys.modules, "cuquantum", cuquantum)
    monkeypatch.setitem(sys.modules, "cuquantum.bindings", cuquantum_bindings)

    module = _load_module("_test_cuquantum_compat_bindings")

    assert module.cusv is bindings


def test_falls_back_to_legacy_custatevec(monkeypatch) -> None:
    legacy = object()

    cuquantum = types.ModuleType("cuquantum")
    cuquantum.__path__ = []
    cuquantum.custatevec = legacy

    monkeypatch.setitem(sys.modules, "cuquantum", cuquantum)
    monkeypatch.delitem(sys.modules, "cuquantum.bindings", raising=False)

    module = _load_module("_test_cuquantum_compat_legacy")

    assert module.cusv is legacy


def test_falls_back_to_legacy_custatevec_when_bindings_lacks_member(monkeypatch) -> None:
    legacy = object()

    cuquantum = types.ModuleType("cuquantum")
    cuquantum.__path__ = []
    cuquantum.custatevec = legacy
    cuquantum_bindings = types.ModuleType("cuquantum.bindings")
    cuquantum_bindings.__path__ = []

    original_import = __import__

    def fake_import(name, globals_=None, locals_=None, fromlist=(), level=0):
        if name == "cuquantum.bindings" and "custatevec" in fromlist:
            msg = "cannot import name 'custatevec' from 'cuquantum.bindings'"
            raise ImportError(msg)
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setitem(sys.modules, "cuquantum", cuquantum)
    monkeypatch.setitem(sys.modules, "cuquantum.bindings", cuquantum_bindings)
    monkeypatch.setattr("builtins.__import__", fake_import)

    module = _load_module("_test_cuquantum_compat_missing_member")

    assert module.cusv is legacy


def test_propagates_non_missing_module_errors(monkeypatch) -> None:
    cuquantum = types.ModuleType("cuquantum")
    cuquantum.__path__ = []  # mark as package for import machinery
    cuquantum_bindings = types.ModuleType("cuquantum.bindings")

    original_import = __import__

    def fake_import(name, globals_=None, locals_=None, fromlist=(), level=0):
        if name == "cuquantum.bindings" and "custatevec" in fromlist:
            msg = "libcustatevec.so: cannot open shared object file"
            raise ModuleNotFoundError(msg, name="libcustatevec.so")
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setitem(sys.modules, "cuquantum", cuquantum)
    monkeypatch.setitem(sys.modules, "cuquantum.bindings", cuquantum_bindings)
    monkeypatch.setattr("builtins.__import__", fake_import)

    try:
        _load_module("_test_cuquantum_compat_import_error")
    except ModuleNotFoundError as exc:
        assert exc.name == "libcustatevec.so"
    else:
        msg = "expected ModuleNotFoundError to propagate"
        raise AssertionError(msg)
