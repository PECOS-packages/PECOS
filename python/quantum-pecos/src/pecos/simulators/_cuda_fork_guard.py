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

"""Detect processes forked after PECOS initialized CUDA.

The Python CUDA simulator wrappers mark their first CUDA call and use this guard
to fail before touching an inherited, unusable CUDA context. Direct users of the
raw ``pecos_rslib_cuda`` extension deliberately bypass this guard; its guided
``Not initialized`` errors remain their backstop.

Registering fork hooks does not load CUDA libraries, create handles, or initialize
the CUDA driver.
"""

from __future__ import annotations

import os
import warnings

__all__ = ["CUDA_FORK_ERROR_MESSAGE", "check_fork_poison", "mark_cuda_initialized"]

CUDA_FORK_ERROR_MESSAGE = (
    "CUDA could not initialize in this process; a common cause is running in a forked child of a process that "
    'already initialized CUDA (CUDA contexts do not survive fork) — use the multiprocessing "spawn" start method.'
)

_state = {
    "cuda_initialized": False,
    "forked_after_cuda_init": False,
    "fork_warning_emitted": False,
}


def _warn_before_fork() -> None:
    """Warn once when this process forks after a PECOS CUDA call."""
    if _state["cuda_initialized"] and not _state["fork_warning_emitted"]:
        _state["fork_warning_emitted"] = True
        warnings.warn(
            "This process is forking after CUDA initialization, so CUDA will be unusable in the forked child. "
            + CUDA_FORK_ERROR_MESSAGE,
            RuntimeWarning,
            stacklevel=2,
        )


def _mark_forked_child() -> None:
    """Record that the child inherited evidence of parent CUDA initialization."""
    if _state["cuda_initialized"]:
        _state["forked_after_cuda_init"] = True


def mark_cuda_initialized() -> None:
    """Record that this process is about to make a real CUDA call."""
    _state["cuda_initialized"] = True


def check_fork_poison() -> None:
    """Fail before CUDA use in a child forked after parent CUDA initialization."""
    if _state["forked_after_cuda_init"]:
        raise RuntimeError(CUDA_FORK_ERROR_MESSAGE)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(before=_warn_before_fork)
    os.register_at_fork(after_in_child=_mark_forked_child)
