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

"""GPU-free constructor wiring tests for the pytket MPS CUDA simulator."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest
from pecos.simulators import _cuda_fork_guard as guard

_MODULE_PATH = Path(__file__).resolve().parents[3] / "src/pecos/simulators/mps_pytket/state.py"


@pytest.fixture(autouse=True)
def _reset_guard_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep synthetic CUDA initialization state isolated between tests."""
    monkeypatch.setattr(
        guard,
        "_state",
        {"cuda_initialized": False, "forked_after_cuda_init": False, "fork_warning_emitted": False},
    )


def _load_mps_module(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    """Load MPS state with stub pytket and cuTensorNet modules."""
    pytket = types.ModuleType("pytket")
    pytket.__path__ = []
    pytket.Qubit = Mock(side_effect=lambda index: index)
    extensions = types.ModuleType("pytket.extensions")
    extensions.__path__ = []
    cutensornet = types.ModuleType("pytket.extensions.cutensornet")
    cutensornet.__path__ = []
    structured_state = types.ModuleType("pytket.extensions.cutensornet.structured_state")

    config = Mock(return_value=types.SimpleNamespace(_complex_t=complex))
    handle = types.SimpleNamespace(destroy=Mock())
    handle_factory = Mock(return_value=handle)
    logger = types.SimpleNamespace(info=Mock())
    mps_factory = Mock(return_value=types.SimpleNamespace(_logger=logger))
    structured_state.Config = config
    structured_state.CuTensorNetHandle = handle_factory
    structured_state.MPSxGate = mps_factory

    mps_package = types.ModuleType("pecos.simulators.mps_pytket")
    mps_package.__path__ = []
    bindings = types.ModuleType("pecos.simulators.mps_pytket.bindings")
    bindings.gate_dict = {}
    mps_package.bindings = bindings

    for name, module in {
        "pytket": pytket,
        "pytket.extensions": extensions,
        "pytket.extensions.cutensornet": cutensornet,
        "pytket.extensions.cutensornet.structured_state": structured_state,
        "pecos.simulators.mps_pytket": mps_package,
        "pecos.simulators.mps_pytket.bindings": bindings,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("_test_mps_cuda_fork_guard", _MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MPS.__del__ = Mock()
    return types.SimpleNamespace(
        module=module,
        config=config,
        handle_factory=handle_factory,
        mps_factory=mps_factory,
        qubit=pytket.Qubit,
    )


def test_mps_constructor_marks_cuda_initialized(monkeypatch) -> None:
    """Successful handle construction records the MPS wrapper's CUDA call."""
    stubs = _load_mps_module(monkeypatch)

    sim = stubs.module.MPS(2)

    assert vars(guard)["_state"]["cuda_initialized"]
    stubs.handle_factory.assert_called_once_with()
    stubs.mps_factory.assert_called_once()
    assert sim.libhandle is stubs.handle_factory.return_value


def test_mps_constructor_checks_fork_poison_before_cuda(monkeypatch) -> None:
    """A poisoned child fails before MPS configuration or CUDA handle creation."""
    stubs = _load_mps_module(monkeypatch)
    vars(guard)["_state"]["forked_after_cuda_init"] = True

    with pytest.raises(RuntimeError) as exc_info:
        stubs.module.MPS(2)

    assert str(exc_info.value) == guard.CUDA_FORK_ERROR_MESSAGE
    stubs.config.assert_not_called()
    stubs.handle_factory.assert_not_called()
    stubs.mps_factory.assert_not_called()
    stubs.qubit.assert_not_called()
