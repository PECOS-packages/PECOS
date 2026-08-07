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

"""Import-isolation tests for optional native extensions.

Each test proves, in one fresh interpreter: (1) ``import pecos`` leaves the
extension out of ``sys.modules``; (2) the native binary genuinely exists and
imports afterwards -- checking the wrapper package alone would let a
wrapper-without-binary installation pass vacuously; (3) the lazily resolved
public symbols work.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap

import pytest

_LLVM_SCRIPT = textwrap.dedent(
    """
    import sys

    import pecos

    loaded = sorted(name for name in sys.modules if "pecos_rslib_llvm" in name)
    if loaded:
        raise AssertionError(f"import pecos loaded pecos_rslib_llvm eagerly: {loaded}")

    # Prove the native binary itself is installed and importable, not just the
    # pure-Python wrapper package.
    import pecos_rslib_llvm.pecos_rslib_llvm  # noqa: F401

    # Prove the lazy public symbols resolve to working objects.
    from pecos.engines import compile_hugr_to_qis, get_compilation_backends

    if not callable(compile_hugr_to_qis):
        raise AssertionError("compile_hugr_to_qis did not resolve to a callable")
    backends = get_compilation_backends()
    if "backends" not in backends:
        raise AssertionError(f"unexpected get_compilation_backends() result: {backends!r}")
    """,
)

_CUDA_SCRIPT = textwrap.dedent(
    """
    import sys

    import pecos

    loaded = sorted(name for name in sys.modules if "pecos_rslib_cuda" in name)
    if loaded:
        raise AssertionError(f"import pecos loaded pecos_rslib_cuda eagerly: {loaded}")

    import pecos_rslib_cuda.pecos_rslib_cuda  # noqa: F401

    from pecos.simulators import CudaStateVec

    if CudaStateVec is None:
        raise AssertionError("CudaStateVec resolved to None despite pecos-rslib-cuda being installed")
    """,
)


def _run_fresh_interpreter(script: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"fresh interpreter failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def test_import_pecos_does_not_load_llvm_extension() -> None:
    """A fresh ``import pecos`` leaves the (always-installed) LLVM extension unloaded."""
    _run_fresh_interpreter(_LLVM_SCRIPT)


def test_import_pecos_does_not_load_cuda_extension() -> None:
    """A fresh ``import pecos`` leaves the CUDA extension unloaded, when it is installed."""
    if importlib.util.find_spec("pecos_rslib_cuda") is None:
        pytest.skip("pecos-rslib-cuda is not installed; the CUDA laziness guard cannot run")
    _run_fresh_interpreter(_CUDA_SCRIPT)
