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


def _inject_optional_dependency_stubs(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Inject the subset of CuPy and cuQuantum metadata used before allocation."""
    cp = types.ModuleType("cupy")
    cuda = types.ModuleType("cupy.cuda")
    runtime = types.ModuleType("cupy.cuda.runtime")
    driver = types.ModuleType("cupy.cuda.driver")
    runtime.CUDARuntimeError = _StubRuntimeError
    runtime.runtimeGetVersion = Mock(return_value=12000)
    driver.CUDADriverError = _StubDriverError
    cuda.runtime = runtime
    cuda.driver = driver
    cuda.Device = Mock()
    cp.cuda = cuda
    cp.complex128 = object()
    cp.zeros = Mock(return_value=[0, 0])

    monkeypatch.setitem(sys.modules, "cupy", cp)
    monkeypatch.setitem(sys.modules, "cupy.cuda", cuda)
    monkeypatch.setitem(sys.modules, "cupy.cuda.runtime", runtime)
    monkeypatch.setitem(sys.modules, "cupy.cuda.driver", driver)
    monkeypatch.setattr(custatevec_state, "cp", cp)
    monkeypatch.setattr(custatevec_state, "require_custatevec", Mock())
    monkeypatch.setattr(custatevec_state, "cudaDataType", types.SimpleNamespace(CUDA_C_64F=object()))
    monkeypatch.setattr(custatevec_state, "ComputeType", types.SimpleNamespace(COMPUTE_64F=object()))
    return cp


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
