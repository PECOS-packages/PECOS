# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""QASM regression test configuration and fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pecos.slr import SlrConverter

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def compare_qasm() -> Callable[..., None]:
    """Pytest fixture for comparing QASM output against regression files."""

    def _compare_qasm(
        block: object,
        *params: object,
        directory: Path | None = None,
        filename: str | None = None,
    ) -> None:
        if directory is None:
            directory = Path(__file__).parent

        if filename is None:
            filename = str(type(block))[8:-2]

        if params:
            params = [str(p) for p in params]
            params = "_".join(params)
            filename = f"{filename}_{params}"

        filename = f"{filename}.qasm"
        file_dir = directory / "regression_qasm" / filename

        with Path(file_dir).open(encoding="utf-8") as file:
            qasm1 = file.read()

        qasm1 = qasm1.strip()
        # TODO: Fix this... this is kinda hacky
        if (
            hasattr(block, "qargs")
            and hasattr(block, "params")
            and hasattr(block, "sym")
        ):
            qasm2 = block.gen("qasm").strip()
        elif hasattr(block, "gen"):
            qasm2 = block.gen("qasm", add_versions=False).strip()
        else:
            qasm2 = SlrConverter(block).qasm(add_versions=False).strip()

        assert qasm1 == qasm2

    return _compare_qasm
