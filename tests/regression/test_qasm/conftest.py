from __future__ import annotations

from pathlib import Path

import pytest

from pecos.slr import SlrConverter
from pecos.slr.main import Main


@pytest.fixture()
def compare_qasm():
    def _compare_qasm(slr, *params, directory: Path | None = None, filename: str | None = None):

        if directory is None:
            directory = Path(__file__).parent

        if filename is None:
            filename = str(type(slr))[8:-2]

        if params:
            params = [str(p) for p in params]
            params = "_".join(params)
            filename = f"{filename}_{params}"

        filename = f"{filename}.qasm"
        file_dir = directory / "regression_qasm" / filename

        with Path(file_dir).open() as file:
            qasm1 = file.read()

        qasm1 = qasm1.strip()

        # wrap the gate in an actual block
        # so that we can use the qasm() method
        skip_headers = False
        if not isinstance(slr, Main):
            main = Main()
            main.extend(slr)
            skip_headers = True
        else:
            main = slr

        qasm2 = SlrConverter(main).qasm(skip_headers=skip_headers).strip()

        # Old version
        # if hasattr(gate, "gen"):
        #     qasm2 = gate.gen("qasm").strip()
        # else:
        #     qasm2 = gate.qasm().strip()

        assert qasm1 == qasm2

    return _compare_qasm
