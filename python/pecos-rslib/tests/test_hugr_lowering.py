"""The bindings wheel owns HUGR lowering without requiring quantum-pecos."""

import re
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


class _FailingRepr:
    def __repr__(self) -> str:
        msg = "repr must not be called unguarded"
        raise RuntimeError(msg)


class _HostileValue(_FailingRepr):
    @property
    def __class__(self) -> type:
        msg = "class lookup must not be called unguarded"
        raise RuntimeError(msg)


@pytest.fixture
def default_int_digit_limit():
    previous = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(4300)
    yield
    sys.set_int_max_str_digits(previous)


# reprlib names a failing-repr object by class; anything it cannot format at all gets
# the placeholder. Python 3.14+ reprlib describes an over-long int itself.
UNPRINTABLE_CASES = [
    pytest.param(_FailingRepr(), r"<_FailingRepr instance at 0x[0-9a-f]+>", id="failing-repr"),
    pytest.param(_HostileValue(), re.escape("<unprintable value>"), id="hostile-object"),
    pytest.param(
        10**5000,
        r"(?:<unprintable value>|<int instance with roughly 5001 digits \(limit at 4300\) at 0x[0-9a-f]+>)",
        id="huge-int",
    ),
]


@pytest.mark.usefixtures("default_int_digit_limit")
@pytest.mark.parametrize(("value", "described"), UNPRINTABLE_CASES)
def test_rejected_option_errors_survive_unprintable_values(
    monkeypatch: pytest.MonkeyPatch,
    value: object,
    described: str,
) -> None:
    monkeypatch.setitem(sys.modules, "selene_hugr_qis_compiler", None)
    checks = [
        ("opt_level must be one of 0, 1, 2, 3; got ", lambda: hugr_lowering.compile_hugr_to_qis(b"", opt_level=value)),
        ("platform must be 'helios' or 'sol'; got ", lambda: hugr_lowering.compile_hugr_to_qis(b"", platform=value)),
    ]
    for prefix, check in checks:
        with pytest.raises(ValueError, match=f"^{re.escape(prefix)}{described}$"):
            check()


@pytest.mark.parametrize("value", [{}, 1, True, "modules"])
def test_lowering_requires_wasm_options(value):
    pytest.importorskip("pecos_selene_wasm")
    message = f"wasm must be a WasmOptions; got {type(value).__name__}; pass wasm=WasmOptions(modules)"
    with pytest.raises(TypeError, match=f"^{re.escape(message)}$"):
        hugr_lowering.compile_hugr_to_qis(b"unused", wasm=value)


def test_wasm_import_order_and_guidance(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "selene_hugr_qis_compiler", None)
    monkeypatch.setitem(sys.modules, "pecos_selene_wasm", None)
    with pytest.raises(ImportError) as error:
        hugr_lowering.compile_hugr_to_qis(b"unused", wasm={})
    assert str(error.value) == "HUGR -> QIS lowering requires the selene-hugr-qis-compiler package"
    monkeypatch.setitem(sys.modules, "selene_hugr_qis_compiler", SimpleNamespace(compile_to_llvm_ir=None))
    with pytest.raises(ImportError) as error:
        hugr_lowering.compile_hugr_to_qis(b"unused", wasm={})
    assert str(error.value) == "Guppy WASM lowering requires the pecos-selene-wasm package (quantum-pecos[wasm])"


def test_lowering_forwards_options_without_revalidation(monkeypatch):
    plugin = pytest.importorskip("pecos_selene_wasm")
    options = plugin.WasmOptions(
        {"m": b"payload"}, int_width="auto", timeout_seconds=2.5, auto_init=False, state_scope="worker", async_workers=2
    )
    calls = []

    def record(program, wasm, **kwargs):
        assert wasm is options
        calls.append((program, kwargs))
        return "; compiled WASM"

    def reject_revalidation(_self):
        pytest.fail("options were reconstructed")

    monkeypatch.setattr(plugin, "compile_wasm", record)
    monkeypatch.setattr(plugin.WasmOptions, "__post_init__", reject_revalidation)
    assert (
        hugr_lowering.compile_hugr_to_qis(b"package", wasm=options, platform="sol", opt_level=3, emit_debug=True)
        == "; compiled WASM"
    )
    assert calls == [(b"package", {"platform": "sol", "opt_level": 3, "emit_debug": True})]
