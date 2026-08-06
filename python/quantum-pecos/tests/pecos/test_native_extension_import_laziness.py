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

"""Import-isolation tests for optional native extensions."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap

import pytest

_SCRIPT = textwrap.dedent(
    """
    import importlib.util
    import sys

    import pecos

    native_name = {native_name!r}
    if importlib.util.find_spec(native_name) is None:
        # The extension is absent entirely, so its non-appearance in
        # sys.modules would prove nothing about import laziness.
        raise AssertionError(f"{{native_name}} is not installed; laziness cannot be tested")
    loaded = sorted(name for name in sys.modules if native_name in name)
    if loaded:
        raise AssertionError(f"import pecos loaded {{native_name}} eagerly: {{loaded}}")
    """,
)


def _assert_import_pecos_leaves_extension_unloaded(native_name: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT.format(native_name=native_name)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"fresh interpreter failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def test_import_pecos_does_not_load_llvm_extension() -> None:
    """A fresh ``import pecos`` leaves the (always-installed) LLVM extension unloaded."""
    _assert_import_pecos_leaves_extension_unloaded("pecos_rslib_llvm")


def test_import_pecos_does_not_load_cuda_extension() -> None:
    """A fresh ``import pecos`` leaves the CUDA extension unloaded, when it is installed."""
    if importlib.util.find_spec("pecos_rslib_cuda") is None:
        pytest.skip("pecos-rslib-cuda is not installed; the CUDA laziness guard cannot run")
    _assert_import_pecos_leaves_extension_unloaded("pecos_rslib_cuda")
