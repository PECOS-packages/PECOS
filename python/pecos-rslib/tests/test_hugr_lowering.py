"""The bindings wheel owns HUGR lowering without requiring quantum-pecos."""

import re
import sys

import pytest
from pecos_rslib import hugr_lowering

# Every "requires explicit wasm_modules" error ends with this fix.
PASS_MODULES = re.escape("pass wasm_modules={module_name: wasm_bytes}")


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


@pytest.mark.parametrize("width", [16, 32.0, True, "32", "AUTO", None])
def test_invalid_wasm_int_width_names_the_value(width: object) -> None:
    with pytest.raises(ValueError, match=r"^wasm_int_width must be 32 or 64, or 'auto'; got ") as error:
        hugr_lowering.validate_wasm_options({"m": b""}, width)
    assert str(error.value).endswith(repr(width))


def test_nondefault_wasm_int_width_requires_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "selene_hugr_qis_compiler", None)
    with pytest.raises(ValueError, match=rf"^wasm_int_width=32 requires explicit wasm_modules; {PASS_MODULES}$"):
        hugr_lowering.compile_hugr_to_qis(b"unused", wasm_int_width=32)


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
        ("wasm_int_width must be 32 or 64, or 'auto'; got ", lambda: hugr_lowering.validate_wasm_options({}, value)),
        ("wasm_auto_init must be True or False; got ", lambda: hugr_lowering.validate_wasm_auto_init(value, {})),
        (
            "wasm_timeout_seconds must be a positive finite number of seconds; got ",
            lambda: hugr_lowering.validate_wasm_timeout_seconds(value, {}),
        ),
        ("opt_level must be one of 0, 1, 2, 3; got ", lambda: hugr_lowering.compile_hugr_to_qis(b"", opt_level=value)),
        ("platform must be 'helios' or 'sol'; got ", lambda: hugr_lowering.compile_hugr_to_qis(b"", platform=value)),
    ]
    for prefix, check in checks:
        with pytest.raises(ValueError, match=f"^{re.escape(prefix)}{described}$"):
            check()
