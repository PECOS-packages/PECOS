"""The bindings wheel owns HUGR lowering without requiring quantum-pecos."""

import sys

import pytest
from pecos_rslib import hugr_lowering


def test_missing_compiler_names_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "selene_hugr_qis_compiler", None)
    with pytest.raises(ImportError, match="selene-hugr-qis-compiler") as error:
        hugr_lowering.compile_hugr_to_qis(b"unused")
    assert str(error.value) == "HUGR -> QIS lowering requires the selene-hugr-qis-compiler package"


@pytest.mark.parametrize("opt_level", [-1, 4, 99, None, "2", 2.0, True, []])
def test_invalid_opt_level_is_rejected_before_compiler_import(
    monkeypatch: pytest.MonkeyPatch, opt_level: object
) -> None:
    monkeypatch.setitem(sys.modules, "selene_hugr_qis_compiler", None)
    with pytest.raises(ValueError, match="opt_level must be one of 0, 1, 2, 3"):
        hugr_lowering.compile_hugr_to_qis(b"unused", opt_level=opt_level)


@pytest.mark.parametrize("platform", ["native", "", None, 0, []])
def test_invalid_platform_is_rejected_before_compiler_import(monkeypatch: pytest.MonkeyPatch, platform: object) -> None:
    monkeypatch.setitem(sys.modules, "selene_hugr_qis_compiler", None)
    with pytest.raises(ValueError, match="platform must be 'helios' or 'sol'"):
        hugr_lowering.compile_hugr_to_qis(b"unused", platform=platform)


@pytest.mark.parametrize("opt_level", [0, 1, 2, 3])
@pytest.mark.parametrize("platform", ["helios", "sol"])
def test_valid_compiler_options_are_forwarded(monkeypatch: pytest.MonkeyPatch, opt_level: int, platform: str) -> None:
    from types import SimpleNamespace

    calls = []

    def compile_recorded(data: bytes, **options: object) -> str:
        calls.append((data, options))
        return "; compiled"

    monkeypatch.setitem(sys.modules, "selene_hugr_qis_compiler", SimpleNamespace(compile_to_llvm_ir=compile_recorded))
    assert hugr_lowering.compile_hugr_to_qis(b"package", opt_level=opt_level, platform=platform) == "; compiled"
    assert calls == [(b"package", {"opt_level": opt_level, "platform": platform, "emit_debug": False})]
