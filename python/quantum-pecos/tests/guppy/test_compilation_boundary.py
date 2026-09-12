"""The Python boundary owns compiler options, normalization and error propagation."""

import pytest
import selene_hugr_qis_compiler
from guppylang import guppy
from guppylang.std.builtins import result
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


def test_hosted_operation_traces_match_rust_compiler() -> None:
    """Classical lowering and helper calls must preserve the complete QIS trace."""
    from guppylang.std.builtins import owned
    from guppylang.std.quantum import cx
    from pecos import Qis, capture_qis_operation_trace
    from pecos_rslib_llvm import compile_hugr_to_qis as rust_compile

    @guppy.declare
    def pecos_qis_trace_metadata_hugr(key: str, value: str) -> None: ...

    @guppy.declare
    def pecos_qis_trace_metadata_qubit_hugr(q: qubit @ owned, key: str, value: str) -> qubit: ...

    @guppy.declare
    def pecos_qis_runtime_barrier_qubit_hugr(q: qubit @ owned) -> qubit: ...

    @guppy.declare
    def pecos_qis_runtime_barrier_qubits2_hugr(a: qubit @ owned, b: qubit @ owned) -> tuple[qubit, qubit]: ...

    @guppy
    def hosted_program() -> None:
        a, b = qubit(), qubit()
        a, b = pecos_qis_runtime_barrier_qubits2_hugr(a, b)
        a = pecos_qis_trace_metadata_qubit_hugr(a, "host_id", "pair:0")
        a = pecos_qis_trace_metadata_qubit_hugr(a, "local_role", "basis_prefix")
        h(a)
        a = pecos_qis_runtime_barrier_qubit_hugr(a)
        pecos_qis_trace_metadata_hugr("host_id", "pair:0")
        cx(a, b)
        bit = measure(a).read()
        count = 1 if bit else 3
        for i in range(count):
            if i % 2 == 0:
                h(b)
        result("count", count)
        result("outcome", measure(b).read())

    data = hosted_program.compile().to_bytes()
    boundary_trace = capture_qis_operation_trace(Qis(compilation_pipeline.compile_hugr_to_qis(data)), 2, seed=42)
    rust_trace = capture_qis_operation_trace(Qis(rust_compile(data)), 2, seed=42)
    assert boundary_trace
    # Each engine instance has its own identifier; compare every payload field.
    boundary_payload = [
        {key: value for key, value in chunk.items() if key != "engine_trace_id"} for chunk in boundary_trace
    ]
    rust_payload = [{key: value for key, value in chunk.items() if key != "engine_trace_id"} for chunk in rust_trace]
    assert boundary_payload == rust_payload
    assert any("pair:0" in str(chunk) for chunk in boundary_trace)
