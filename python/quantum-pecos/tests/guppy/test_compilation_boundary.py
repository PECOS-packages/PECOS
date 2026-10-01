"""The Python boundary owns compiler options, normalization and error propagation."""

import pytest
import selene_hugr_qis_compiler
from guppylang import guppy
from guppylang.std.quantum import h, measure, qubit
from pecos import compilation_pipeline, execute_llvm


@guppy
def coin() -> bool:
    q = qubit()
    h(q)
    return measure(q).read()


@pytest.mark.parametrize("text_envelope", [False, True])
def test_boundary_accepts_both_envelopes(text_envelope: bool) -> None:
    package = coin.compile()
    data = package.to_str().encode() if text_envelope else package.to_bytes()
    assert "@qmain(" in compilation_pipeline.compile_hugr_to_qis(data)


@pytest.mark.parametrize(
    ("data", "error_type"),
    [(b"invalid", selene_hugr_qis_compiler.HugrReadError), ("invalid", TypeError)],
)
def test_external_errors_propagate(data: object, error_type: type[Exception]) -> None:
    with pytest.raises(error_type):
        compilation_pipeline.compile_hugr_to_qis(data)


def test_boundary_forwards_options_and_normalizes(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = "pecos_qis_runtime_barrier_qubit_hugr"
    calls = []

    def compile_recorded(data: bytes, **options: object) -> str:
        calls.append((data, options))
        return f"declare i64 @__hugr__.__main__.{helper}.14(i64)\n"

    monkeypatch.setattr(selene_hugr_qis_compiler, "compile_to_llvm_ir", compile_recorded)
    actual = compilation_pipeline.compile_hugr_to_qis(b"package", platform="sol", opt_level=0, emit_debug=True)
    assert actual == f"declare i64 @{helper}(i64)\n"
    assert calls == [(b"package", {"platform": "sol", "opt_level": 0, "emit_debug": True})]


def test_guppy_debug_option_reaches_external_compiler(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    original = selene_hugr_qis_compiler.compile_to_llvm_ir

    def compile_recorded(data: bytes, **options: object) -> str:
        calls.append(options)
        return original(data, **options)

    monkeypatch.setattr(selene_hugr_qis_compiler, "compile_to_llvm_ir", compile_recorded)
    assert "@qmain(" in compilation_pipeline.compile_guppy_to_llvm(coin, emit_debug=True)
    assert calls == [{"platform": "helios", "opt_level": 2, "emit_debug": True}]


def test_compiler_exception_identity_is_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    error = selene_hugr_qis_compiler.HugrReadError("invalid package")

    def fail(*_args: object, **_kwargs: object) -> str:
        raise error

    monkeypatch.setattr(selene_hugr_qis_compiler, "compile_to_llvm_ir", fail)
    for call in (
        lambda: compilation_pipeline.compile_hugr_to_qis(b"bad"),
        lambda: execute_llvm.compile_module_to_string(b"bad"),
    ):
        with pytest.raises(selene_hugr_qis_compiler.HugrReadError) as caught:
            call()
        assert caught.value is error


def test_boundary_is_owned_by_bindings() -> None:
    from pecos_rslib.hugr_lowering import compile_hugr_to_qis

    assert compilation_pipeline.compile_hugr_to_qis is compile_hugr_to_qis
