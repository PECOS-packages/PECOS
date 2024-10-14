from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def compare_qasm():
    def _compare_qasm(block, *params, directory: Path | None = None, filename: str | None = None):

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

        with Path(file_dir).open() as file:
            qasm1 = file.read()

        qasm1 = qasm1.strip()

        if hasattr(block, "gen"):
            qasm2 = block.gen("qasm").strip()
        else:
            qasm2 = block.qasm().strip()

        assert qasm1 == qasm2

    return _compare_qasm
