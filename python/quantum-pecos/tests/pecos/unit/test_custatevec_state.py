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

"""Tests for actionable CUDA initialization errors from the legacy CuPy simulator.

The tests inject stub optional dependencies so the error handling remains covered
whether or not CuPy and cuQuantum are installed in the test environment.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import Mock

import pytest
from pecos.simulators import _cuda_fork_guard as guard
from pecos.simulators.custatevec import state as custatevec_state


class _StubCudaStatusError(RuntimeError):
    """CuPy-compatible CUDA error carrying a numeric status."""

    def __init__(self, status: int) -> None:
        super().__init__(status)
        self.status = status


class _StubRuntimeError(_StubCudaStatusError):
    """Stand-in for ``cupy.cuda.runtime.CUDARuntimeError``."""


class _StubDriverError(_StubCudaStatusError):
    """Stand-in for ``cupy.cuda.driver.CUDADriverError``."""


@pytest.fixture(autouse=True)
def _reset_guard_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep synthetic failed initialization from marking later tests."""
    monkeypatch.setattr(
        guard,
        "_state",
        {"cuda_initialized": False, "forked_after_cuda_init": False, "fork_warning_emitted": False},
    )


def _inject_optional_dependency_stubs(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Inject the subset of CuPy and cuQuantum metadata used before allocation."""
    cp = types.ModuleType("cupy")
    cuda = types.ModuleType("cupy.cuda")
    runtime = types.ModuleType("cupy.cuda.runtime")
    driver = types.ModuleType("cupy.cuda.driver")
    runtime.CUDARuntimeError = _StubRuntimeError
    runtime.runtimeGetVersion = Mock(return_value=12000)
    runtime.deviceGetDefaultMemPool = Mock(return_value=1)
    runtime.memPoolSetAttribute = Mock()
    runtime.cudaMemPoolAttrReleaseThreshold = object()
    driver.CUDADriverError = _StubDriverError
    cuda.runtime = runtime
    cuda.driver = driver
    cuda.Device = Mock(
        return_value=types.SimpleNamespace(attributes={"MemoryPoolsSupported": True}, id=0),
    )
    cuda.Stream = Mock(return_value=types.SimpleNamespace(ptr=1))
    cp.cuda = cuda
    cp.complex128 = object()
    cp.zeros = Mock(return_value=[0, 0])
    cusv = types.SimpleNamespace(
        create=Mock(return_value=object()),
        set_stream=Mock(),
        set_device_mem_handler=Mock(),
        destroy=Mock(),
    )

    monkeypatch.setitem(sys.modules, "cupy", cp)
    monkeypatch.setitem(sys.modules, "cupy.cuda", cuda)
    monkeypatch.setitem(sys.modules, "cupy.cuda.runtime", runtime)
    monkeypatch.setitem(sys.modules, "cupy.cuda.driver", driver)
    monkeypatch.setattr(custatevec_state, "cp", cp)
    monkeypatch.setattr(custatevec_state, "require_custatevec", Mock())
    monkeypatch.setattr(custatevec_state, "cudaDataType", types.SimpleNamespace(CUDA_C_64F=object()))
    monkeypatch.setattr(custatevec_state, "ComputeType", types.SimpleNamespace(COMPUTE_64F=object()))
    monkeypatch.setattr(custatevec_state, "cusv", cusv)
    return cp


def test_custatevec_constructor_marks_cuda_initialized(monkeypatch) -> None:
    """Successful construction records the wrapper's first CUDA call."""
    cp = _inject_optional_dependency_stubs(monkeypatch)

    sim = custatevec_state.CuStateVec(1)

    assert vars(guard)["_state"]["cuda_initialized"]
    cp.zeros.assert_called_once()
    custatevec_state.cusv.create.assert_called_once()
    assert sim.libhandle is not None
    sim.libhandle = None


def test_custatevec_constructor_checks_fork_poison_before_cuda(monkeypatch) -> None:
    """A poisoned child fails before dependency checks or CUDA calls."""
    cp = _inject_optional_dependency_stubs(monkeypatch)
    vars(guard)["_state"]["forked_after_cuda_init"] = True

    with pytest.raises(RuntimeError) as exc_info:
        custatevec_state.CuStateVec(1)

    assert str(exc_info.value) == guard.CUDA_FORK_ERROR_MESSAGE
    custatevec_state.require_custatevec.assert_not_called()
    cp.zeros.assert_not_called()
    cp.cuda.runtime.runtimeGetVersion.assert_not_called()
    cp.cuda.Device.assert_not_called()
    custatevec_state.cusv.create.assert_not_called()


def test_custatevec_reset_checks_fork_poison(monkeypatch) -> None:
    """An inherited simulator fails before reset touches its CUDA state."""
    _inject_optional_dependency_stubs(monkeypatch)
    sim = custatevec_state.CuStateVec(1)
    vars(guard)["_state"]["forked_after_cuda_init"] = True

    with pytest.raises(RuntimeError) as exc_info:
        sim.reset()

    assert str(exc_info.value) == guard.CUDA_FORK_ERROR_MESSAGE
    sim.libhandle = None


def test_cuda_initialization_errors_explain_fork_and_preserve_cause(monkeypatch) -> None:
    """Translate the runtime and driver initialization statuses from CuPy calls."""
    cp = _inject_optional_dependency_stubs(monkeypatch)

    runtime_error = cp.cuda.runtime.CUDARuntimeError(3)
    cp.zeros.side_effect = runtime_error

    with pytest.raises(RuntimeError, match='multiprocessing "spawn" start method') as exc_info:
        custatevec_state.CuStateVec(1)

    assert type(exc_info.value) is RuntimeError
    assert "forked child" in str(exc_info.value)
    assert exc_info.value.__cause__ is runtime_error

    driver_error = cp.cuda.driver.CUDADriverError(3)
    cp.zeros.side_effect = None
    cp.cuda.Device.side_effect = driver_error

    with pytest.raises(RuntimeError, match='multiprocessing "spawn" start method') as exc_info:
        custatevec_state.CuStateVec(1)

    assert type(exc_info.value) is RuntimeError
    assert "forked child" in str(exc_info.value)
    assert exc_info.value.__cause__ is driver_error


def test_other_cuda_runtime_status_propagates_unchanged(monkeypatch) -> None:
    """Do not translate CUDARuntimeError statuses unrelated to initialization."""
    cp = _inject_optional_dependency_stubs(monkeypatch)
    cuda_error = cp.cuda.runtime.CUDARuntimeError(1)  # cudaErrorInvalidValue
    cp.zeros.side_effect = cuda_error

    with pytest.raises(cp.cuda.runtime.CUDARuntimeError) as exc_info:
        custatevec_state.CuStateVec(1)

    assert exc_info.value is cuda_error
