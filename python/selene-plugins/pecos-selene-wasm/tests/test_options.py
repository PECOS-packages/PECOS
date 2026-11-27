"""The shared immutable WASM configuration owns validation and value semantics."""

import copy
import hashlib
import json
import pickle
import re
import reprlib
import struct
import sys
from collections import UserDict
from collections.abc import Hashable
from dataclasses import FrozenInstanceError, dataclass, replace
from pathlib import Path

import pytest
from guppylang import guppy
from guppylang.std.builtins import nat, result
from guppylang_internals.decorator import wasm, wasm_module
from hugr.envelope import EnvelopeConfig, EnvelopeFormat
from hugr.package import Package
from pecos_selene_wasm import WasmOptions, build, compile_guppy, compile_wasm

RULES = {
    "int_width": "must be 32 or 64, or 'auto'",
    "timeout_seconds": "must be a positive finite number of seconds",
    "auto_init": "must be True or False",
    "state_scope": "must be 'shot' or 'worker'",
    "async_workers": "must be an integer between 0 and 64",
}


class IntSubclass(int):
    pass


class FloatSubclass(float):
    pass


class StrSubclass(str):
    __slots__ = ()


class BytesSubclass(bytes):
    pass


@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("int_width", [32, 64, "auto"]),
        ("timeout_seconds", [1, 2, 0.05, 1.0, float.fromhex("0x0.0000000000001p-1022"), sys.float_info.max]),
        ("auto_init", [True, False]),
        ("state_scope", ["shot", "worker"]),
        ("async_workers", range(65)),
    ],
)
def test_scalar_acceptance(field, values):
    for value in values:
        options = WasmOptions({}, **{field: value})
        assert getattr(options, field) == value
        if field == "timeout_seconds":
            assert type(options.timeout_seconds) is float


@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("int_width", [16, 32.0, 64.0, True, False, "32", "AUTO", None, [], IntSubclass(32), StrSubclass("auto")]),
        (
            "timeout_seconds",
            [
                True,
                False,
                "2",
                None,
                0,
                -1,
                float("nan"),
                float("inf"),
                -float("inf"),
                10**400,
                IntSubclass(1),
                FloatSubclass(1),
            ],
        ),
        ("auto_init", [0, 1, None, "false", [], IntSubclass(1)]),
        ("state_scope", [None, 1, True, "batch", "WORKER", [], StrSubclass("shot")]),
        ("async_workers", [-1, 65, True, False, 2.0, "2", None, [], IntSubclass(2)]),
    ],
)
def test_scalar_rejection(field, values):
    for value in values:
        message = f"WasmOptions.{field} {RULES[field]}; got {reprlib.repr(value)}"
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            WasmOptions({}, **{field: value})


def test_constructor_shape_and_defaults():
    options = WasmOptions(modules={})
    assert (
        options.int_width,
        options.timeout_seconds,
        options.auto_init,
        options.state_scope,
        options.async_workers,
    ) == (64, 1.0, True, "shot", 0)
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'modules'"):
        WasmOptions()
    with pytest.raises(TypeError, match="positional arguments"):
        WasmOptions({}, 32)


@pytest.mark.parametrize("value", [None, [], [("m", b"")], "module", b"module"])
def test_modules_require_mapping(value):
    message = f"WasmOptions.modules must be a Mapping; got {type(value).__name__}: {reprlib.repr(value)}"
    with pytest.raises(TypeError, match=f"^{re.escape(message)}$"):
        WasmOptions(value)


@pytest.mark.parametrize("key", [1, None, b"m", ("m",)])
def test_module_keys_name_the_key_and_type(key):
    message = f"WasmOptions.modules key {reprlib.repr(key)} must be a str; got {type(key).__name__}"
    with pytest.raises(TypeError, match=f"^{re.escape(message)}$"):
        WasmOptions({key: b""})


@pytest.mark.parametrize("value", [None, "m.wasm", Path("m.wasm"), bytearray(b"wasm"), memoryview(b"wasm"), 7])
def test_module_values_name_the_key_and_type(value):
    message = (
        "WasmOptions.modules['unused'] must be explicit bytes, not paths; "
        f"got {type(value).__name__}: {reprlib.repr(value)}"
    )
    with pytest.raises(TypeError, match=f"^{re.escape(message)}$"):
        WasmOptions({"valid": b"", "unused": value})


def test_modules_accept_mapping_and_bytes_subclasses():
    modules = UserDict({StrSubclass("m"): BytesSubclass(b"payload")})
    assert WasmOptions(modules).modules == {"m": b"payload"}


def test_modules_are_an_immutable_snapshot():
    source = {"m": b"original"}
    options = WasmOptions(source)
    source["m"] = b"changed"
    source["new"] = b"added"
    assert options.modules == {"m": b"original"}
    with pytest.raises(TypeError, match="does not support item assignment"):
        options.modules["m"] = b"changed"
    with pytest.raises(FrozenInstanceError, match="cannot assign to field 'modules'"):
        options.modules = {}
    with pytest.raises(FrozenInstanceError, match="cannot assign to field 'int_width'"):
        options.int_width = 32


def test_replace_revalidates():
    options = WasmOptions({"m": b""})
    assert replace(options, int_width=32) == WasmOptions({"m": b""}, int_width=32)
    with pytest.raises(ValueError, match=r"^WasmOptions\.int_width must be 32 or 64, or 'auto'; got True$"):
        replace(options, int_width=True)
    with pytest.raises(TypeError, match="explicit bytes, not paths"):
        replace(options, modules={"m": "path"})


def test_validation_follows_declaration_order():
    invalid = {
        "modules": [],
        "int_width": 16,
        "timeout_seconds": 0,
        "auto_init": 0,
        "state_scope": "bad",
        "async_workers": 65,
    }
    for field in list(invalid):
        with pytest.raises(TypeError if field == "modules" else ValueError, match=rf"^WasmOptions\.{field} "):
            WasmOptions(**({"modules": {}} | invalid))
        del invalid[field]


def test_value_equality():
    options = WasmOptions({"a": b"one", "b": b"two"}, timeout_seconds=2)
    assert options == WasmOptions({"b": b"two", "a": b"one"}, timeout_seconds=2.0)
    assert options != replace(options, int_width=32)
    assert options != replace(options, modules={"a": b"different"})
    assert options != {}


@pytest.mark.parametrize(
    "roundtrip",
    [copy.deepcopy, lambda obj: pickle.loads(pickle.dumps(obj))],
    ids=["deepcopy", "pickle"],
)
def test_roundtrip_reconstructs_value(roundtrip):
    options = WasmOptions(
        {"m": b"payload"},
        int_width="auto",
        timeout_seconds=2,
        auto_init=False,
        state_scope="worker",
        async_workers=3,
    )
    rebuilt = roundtrip(options)
    assert rebuilt == options
    assert rebuilt is not options
    with pytest.raises(TypeError, match="does not support item assignment"):
        rebuilt.modules["m"] = b"changed"


def test_options_are_unhashable():
    options = WasmOptions({})
    assert WasmOptions.__hash__ is None
    assert not isinstance(options, Hashable)
    with pytest.raises(TypeError, match=r"^unhashable type: 'WasmOptions'$"):
        hash(options)


def test_repr_describes_sizes_without_payloads():
    options = WasmOptions({"large": b"secret" * 1000000, "empty": b""})
    assert repr(options) == (
        "WasmOptions(modules={'large': <6000000 bytes>, 'empty': <0 bytes>}, "
        "int_width=64, timeout_seconds=1.0, auto_init=True, state_scope='shot', async_workers=0)"
    )


@pytest.mark.parametrize("entry", [compile_wasm, build])
@pytest.mark.parametrize("value", [None, {}, 1, object()])
def test_plugin_boundaries_require_options(entry, value, monkeypatch):
    import pecos_selene_wasm

    def unexpected_lowering(*_args, **_kwargs):
        pytest.fail("lowering must not run with invalid options")

    if entry is build:
        monkeypatch.setattr(pecos_selene_wasm, "compile_wasm", unexpected_lowering)
    with pytest.raises(TypeError, match=f"^options must be a WasmOptions; got {type(value).__name__}$"):
        entry(b"invalid HUGR", value)


@pytest.mark.parametrize("name", ["modules", *RULES, *(f"wasm_{field}" for field in ["modules", *RULES])])
def test_build_rejects_misplaced_options(name):
    field = name.removeprefix("wasm_")
    message = f"Pass {field}=... in WasmOptions(...), not as build keyword arguments"
    with pytest.raises(TypeError, match=f"^{re.escape(message)}$"):
        build(b"invalid HUGR", WasmOptions({}), **{name: 10})


def test_build_discovers_option_fields(monkeypatch):
    import pecos_selene_wasm

    @dataclass(frozen=True)
    class ExtendedOptions(WasmOptions):
        future_option: bool = False

    monkeypatch.setattr(pecos_selene_wasm, "WasmOptions", ExtendedOptions)
    with pytest.raises(
        TypeError,
        match=r"^Pass future_option=\.\.\. in WasmOptions\(\.\.\.\), not as build keyword arguments$",
    ):
        build(b"invalid HUGR", ExtendedOptions({}), future_option=True)


EXAMPLES = Path(__file__).resolve().parents[4] / "examples/guppy_wasm"
FIRST = EXAMPLES / "arithmetic.wat"
SECOND = EXAMPLES / "counter.wat"


@wasm_module(str(FIRST))
class Arithmetic:
    @wasm
    def add(self, a: int, b: int) -> int: ...


@wasm_module(str(SECOND))
class Counter:
    @wasm
    def increment(self) -> int: ...


@guppy
def two_modules() -> None:
    a = Arithmetic(nat(0))
    result("sum", a.add(2, 3))
    a.discard()
    b = Counter(nat(1))
    result("count", b.increment())
    b.discard()


@guppy
def ordinary() -> None:
    result("value", 7)


def test_empty_mapping_preserves_lowering_errors():
    options = WasmOptions({})
    with pytest.raises(ValueError, match=r"^Program contains no tket\.wasm operations$"):
        compile_wasm(ordinary, options)
    with pytest.raises(ValueError, match=r"^Supply bytes for WASM module "):
        compile_wasm(two_modules, options)


def test_worker_key_and_registration_follow_reference_order():
    package = Package.from_bytes(compile_guppy(two_modules).to_bytes())
    # The JSON envelope's ten-byte header precedes the serialized document.
    document = json.loads(package.to_bytes(EnvelopeConfig(format=EnvelopeFormat.JSON))[10:])["modules"][0]
    ids = list(
        dict.fromkeys(
            node["v"]["value"]["v"]["module_filename"]
            for node in document["nodes"]
            if node.get("op") == "Const" and node["v"].get("value", {}).get("c") == "ConstWasmModule"
        ),
    )
    assert len(ids) == 2
    modules = {identifier: Path(identifier).read_bytes() for identifier in reversed(ids)}
    modules["unreferenced"] = b"unused payload"
    assert list(modules)[:2] == ids[::-1]
    options = WasmOptions(modules, state_scope="worker", async_workers=2)
    llvm = compile_wasm(package.to_bytes(), options)
    key = hashlib.sha256()
    fields = [package.to_bytes(), repr(("helios", 2, False, 64, 1.0, True, 2)).encode()]
    for identifier in ids:
        fields.extend((identifier.encode(), modules[identifier]))
    for field in fields:
        key.update(struct.pack(">Q", len(field)))
        key.update(field)
    constants = dict(
        re.findall(r'@(pecos_wasm_(?:module|worker_key)_\w+) = private constant \[\d+ x i8\] c"([^"]*)"', llvm),
    )
    decoded = {name: bytes.fromhex(value.replace("\\", "")) for name, value in constants.items()}
    assert [data for name, data in decoded.items() if "worker_key" in name] == [key.digest()]
    for index, identifier in enumerate(ids):
        [name] = [name for name in decoded if name.startswith("pecos_wasm_module_") and name.endswith(f"_{index}")]
        assert decoded[name] == modules[identifier]
        assert f"@pecos_wasm_v1_register(i64 {index}, ptr @{name}," in llvm
    assert re.findall(r"call i32 @pecos_wasm_v1_register\(i64 (\d+),", llvm) == ["0", "1"]
    assert len(decoded) == 3
    reordered = {identifier: modules[identifier] for identifier in ids}
    assert compile_wasm(package.to_bytes(), replace(options, modules=reordered)) == llvm
    reordered["unreferenced"] = b"different unused payload"
    assert compile_wasm(package.to_bytes(), replace(options, modules=reordered)) == llvm


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
@pytest.mark.parametrize(
    ("option", "prefix"),
    [
        ("auto_init", "WasmOptions.auto_init must be True or False; got "),
        ("timeout_seconds", "WasmOptions.timeout_seconds must be a positive finite number of seconds; got "),
        ("int_width", "WasmOptions.int_width must be 32 or 64, or 'auto'; got "),
        ("state_scope", "WasmOptions.state_scope must be 'shot' or 'worker'; got "),
        ("async_workers", "WasmOptions.async_workers must be an integer between 0 and 64; got "),
    ],
)
def test_rejected_option_errors_survive_unprintable_values(value, described, option, prefix):
    with pytest.raises(ValueError, match=f"^{re.escape(prefix)}{described}$"):
        WasmOptions({}, **{option: value})
