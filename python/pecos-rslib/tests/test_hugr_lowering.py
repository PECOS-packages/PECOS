"""The bindings wheel owns HUGR lowering without requiring quantum-pecos."""

import sys

import pytest
from pecos_rslib import hugr_lowering


def test_missing_compiler_names_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "selene_hugr_qis_compiler", None)
    with pytest.raises(ImportError, match="selene-hugr-qis-compiler") as error:
        hugr_lowering.compile_hugr_to_qis(b"unused")
    assert str(error.value) == "HUGR -> QIS lowering requires the selene-hugr-qis-compiler package"
