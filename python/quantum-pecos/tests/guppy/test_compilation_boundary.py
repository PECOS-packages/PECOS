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


def _operation_shape(operation: object) -> tuple:
    """Reduce a trace operation to its kind and integer operands; angles are omitted
    so a different gate decomposition of the same rotation keeps the shape."""
    if isinstance(operation, str):
        return (operation,)
    ((kind, payload),) = operation.items()
    if kind == "Quantum":
        ((gate, operands),) = payload.items()
        operands = operands if isinstance(operands, list) else [operands]
        return (gate, *(value for value in operands if type(value) is int))
    if kind == "TraceMetadata":
        return (kind, payload["qubit"], *sorted(payload["metadata"].items()))
    return (kind, payload["id"])


def test_hosted_operations_keep_their_order_through_lowering() -> None:
    """Barriers and qubit-linked trace metadata stay in place around the gates
    they bracket, and measurement feedback drives the classical loop exactly
    once per shot.

    Qubit-free trace metadata has no dataflow edge to any gate, so lowering may
    place it anywhere; the pinned trace records where it lands (after the CX
    here), not an ordering guarantee."""
    from guppylang.std.builtins import owned, result
    from guppylang.std.quantum import cx
    from pecos import Qis, capture_qis_operation_trace

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
    trace = capture_qis_operation_trace(Qis(compilation_pipeline.compile_hugr_to_qis(data)), 2, seed=42)

    assert [chunk["stage"] for chunk in trace] == [
        "pending_start",
        "pending_continue",
        "named_results",
        "trace_complete",
    ]
    # Seed 42 measures a = 1, so count = 1 and the loop applies h(b) once.
    assert trace[-1]["measurement_results"] == {"0": 1, "1": 1}
    assert [[_operation_shape(op) for op in chunk["operations"]] for chunk in trace[:2]] == [
        [
            ("AllocateQubit", 0),
            ("Reset", 0),
            ("AllocateQubit", 1),
            ("Reset", 1),
            ("Barrier",),
            ("TraceMetadata", 0, ("host_id", "pair:0")),
            ("TraceMetadata", 0, ("local_role", "basis_prefix")),
            ("RXY", 0),
            ("RZ", 0),
            ("Barrier",),
            ("RXY", 1),
            ("RZZ", 0, 1),
            ("RZ", 0),
            ("RXY", 1),
            ("RZ", 1),
            ("TraceMetadata", None, ("host_id", "pair:0")),
            ("AllocateResult", 0),
            ("Measure", 0, 0),
            ("ReleaseQubit", 0),
        ],
        [
            ("RXY", 1),
            ("RZ", 1),
            ("AllocateResult", 1),
            ("Measure", 1, 1),
            ("ReleaseQubit", 1),
        ],
    ]
    assert [named["name"] for named in trace[2]["named_result_traces"]] == ["count", "outcome"]
