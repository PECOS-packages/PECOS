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

"""GPU-free tests for CUDA initialization state inherited across ``fork``."""

from __future__ import annotations

import os
import warnings

import pytest
from pecos.simulators import _cuda_fork_guard as guard

pytestmark = pytest.mark.skipif(not hasattr(os, "fork"), reason="os.fork is unavailable on this platform")


@pytest.fixture(autouse=True)
def _reset_guard_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the guard's process-local state isolated between tests."""
    monkeypatch.setattr(guard, "_cuda_initialized", False)
    monkeypatch.setattr(guard, "_forked_after_cuda_init", False)
    monkeypatch.setattr(guard, "_fork_warning_emitted", False)


def _wait_for_child(pid: int) -> int:
    """Wait for a forked child and return its conventional exit code."""
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status)


def test_marked_parent_poisons_forked_child() -> None:
    """A child inherits the poison marker when its parent marked CUDA use."""
    guard.mark_cuda_initialized()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pid = os.fork()

    if pid == 0:
        try:
            guard.check_fork_poison()
        except RuntimeError as exc:
            os._exit(0 if str(exc) == guard.CUDA_FORK_ERROR_MESSAGE else 2)
        os._exit(1)

    assert _wait_for_child(pid) == 0


def test_unmarked_parent_does_not_poison_forked_child() -> None:
    """Forking before any marked CUDA call leaves the child usable."""
    pid = os.fork()
    if pid == 0:
        try:
            guard.check_fork_poison()
        except RuntimeError:
            os._exit(1)
        os._exit(0)

    assert _wait_for_child(pid) == 0


def test_marked_parent_warns_only_once_before_fork() -> None:
    """Repeated forks after marked CUDA use emit one parent-side warning."""
    guard.mark_cuda_initialized()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        for _ in range(2):
            pid = os.fork()
            if pid == 0:
                os._exit(0)
            assert _wait_for_child(pid) == 0

    fork_warnings = [warning for warning in caught if warning.category is RuntimeWarning]
    assert len(fork_warnings) == 1
    assert "forking after CUDA initialization" in str(fork_warnings[0].message)
    assert guard.CUDA_FORK_ERROR_MESSAGE in str(fork_warnings[0].message)
