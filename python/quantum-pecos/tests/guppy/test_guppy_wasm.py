"""Guppy WASM compilation and real execution through the PECOS sim() stack."""

import importlib.util
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pecos as pc
import pytest
from guppylang import guppy
from guppylang.std.builtins import nat, result
from guppylang.std.quantum import measure, qubit, x
from guppylang_internals.decorator import wasm, wasm_module
from guppylang_internals.error import GuppyError
from hugr import ops
from hugr.package import Package

EXAMPLES = Path(__file__).resolve().parents[4] / "examples/guppy_wasm"


@pytest.fixture(scope="module")
def examples():
    modules = {}
    for name in ("arithmetic", "stateful", "floating_point", "indexed_exports"):
        spec = importlib.util.spec_from_file_location(f"wasm_{name}", EXAMPLES / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


@pytest.fixture
def arithmetic(examples):
    module = pc.WasmForeignObject.from_file(examples["arithmetic"].WASM_PATH)
    module.init()
    try:
        yield module
    finally:
        module.teardown()


@pytest.mark.parametrize(
    ("name", "arguments", "expected"),
    [
        ("add", [2, 3], 5),
        ("add", [0, 0], 0),
        ("add", [-7, 3], -4),
        ("add", [2**40, 3], 2**40 + 3),
        ("multiply", [5, 4], 20),
        ("multiply", [-7, 3], -21),
        ("multiply", [0, 9], 0),
    ],
)
def test_direct_wasm_arithmetic(arithmetic, name, arguments, expected) -> None:
    """Exercise the real exports, including signed and wider-than-i32 values."""
    assert arithmetic.exec(name, arguments) == expected


def test_chained_wasm_calls(arithmetic) -> None:
    total = arithmetic.exec("add", [2, 3])
    assert arithmetic.exec("multiply", [total, 4]) == 20


@pytest.mark.parametrize(
    ("name", "exports"),
    [("addition", ["add"]), ("chained_arithmetic", ["add", "multiply"]), ("conditional_x", ["add"])],
)
def test_guppy_wasm_hugr_roundtrip(examples, name, exports) -> None:
    """PECOS's Guppy wrapper must preserve actual WASM calls and their context."""
    program = pc.Guppy(getattr(examples["arithmetic"], name))
    package = Package.from_bytes(program.hugr_bytes)
    wasm_ops = [
        op
        for graph in package.modules
        for node in graph
        if isinstance(op := graph[node].op, ops.ExtOp) and op.op_def().get_extension().name == "tket.wasm"
    ]
    names = Counter(op.op_def().name for op in wasm_ops)
    assert names == {
        "get_context": 1,
        "lookup_by_name": len(exports),
        "call": len(exports),
        "read_result": len(exports),
        "dispose_context": 1,
    }
    assert [op.args[0].value for op in wasm_ops if op.op_def().name == "lookup_by_name"] == exports


def test_missing_export_is_rejected(examples) -> None:
    with pytest.raises(GuppyError, match="isn't exported"):

        @wasm_module(str(examples["arithmetic"].WASM_PATH))
        class MissingExport:
            @wasm
            def subtract(self, a: int, b: int) -> int: ...


def test_wrong_signature_is_rejected(examples) -> None:
    @wasm_module(str(examples["arithmetic"].WASM_PATH))
    class WrongSignature:
        @wasm
        def add(self, a: int) -> int: ...

    @guppy
    def wrong_call() -> int:
        arithmetic = WrongSignature(nat(0))
        total = arithmetic.add(2)
        arithmetic.discard()
        return total

    with pytest.raises(GuppyError, match="Wasm signature mismatch"):
        wrong_call.compile()


@pytest.mark.parametrize(
    ("name", "label", "expected"),
    [
        ("addition", "sum", 5),
        ("chained_arithmetic", "product", 20),
        ("conditional_x", "measured", 1),
        ("conditional_calls", "value", 6),
    ],
)
@pytest.mark.parametrize("wrapper", ["guppy", "hugr"])
def test_pecos_wasm_execution(examples, name, label, expected, wrapper):
    pytest.importorskip("pecos_selene_wasm", reason="requires quantum-pecos[wasm]")
    example = examples["arithmetic"]
    function = getattr(example, name)
    modules = {str(example.WASM_PATH): example.WASM_PATH.read_bytes()}
    program = (
        pc.Guppy(function, wasm_modules=modules)
        if wrapper == "guppy"
        else pc.Hugr(
            function.compile().to_bytes(),
            wasm_modules=modules,
        )
    )
    data = pc.sim(program).qubits(1).quantum(pc.state_vector()).seed(42).run(4).to_dict()
    assert data[label] == [expected] * 4


def test_pecos_state_and_error_recovery(examples):
    pytest.importorskip("pecos_selene_wasm", reason="requires quantum-pecos[wasm]")
    example = examples["stateful"]
    modules = {str(example.WASM_PATH): example.WASM_PATH.read_bytes()}

    def run(function):
        return pc.sim(pc.Guppy(function, wasm_modules=modules)).qubits(1).quantum(pc.state_vector()).run(4).to_dict()

    for _ in range(2):
        data = run(example.stateful_calls)
        assert data["three"] == [3] * 4
        assert data["four"] == [4] * 4
        assert data["independent"] == [1] * 4
        assert data["cleared"] == [1] * 4
        with pytest.raises(RuntimeError, match=r"WASM|wasm|unreachable|2001"):
            run(example.trapped)
    data = run(example.context_helpers)
    assert data["second"] == [2] * 4
    assert data["third"] == [3] * 4


@pytest.mark.parametrize("backend", ["pecos", "selene"])
def test_example_runs_from_another_directory(tmp_path, backend) -> None:
    pytest.importorskip("pecos_selene_wasm", reason="requires quantum-pecos[wasm]")
    completed = subprocess.run(
        [sys.executable, str(EXAMPLES / "arithmetic.py"), "--run", backend],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert f"{backend}: addition: {{'sum': [5, 5, 5, 5]}}" in completed.stdout
    assert f"{backend}: chained_arithmetic: {{'product': [20, 20, 20, 20]}}" in completed.stdout
    assert f"{backend}: conditional_x: {{'measured': [1, 1, 1, 1]}}" in completed.stdout


@guppy
def control_without_wasm() -> None:
    """Verify both runners and result parsing independently of WASM support."""
    result("sum", 5)
    result("product", 20)
    q = qubit()
    x(q)
    result("measured", measure(q).read())


@pytest.mark.parametrize("backend", ["pecos", "selene"])
def test_runtime_probe_control(tmp_path, backend) -> None:
    """Keep the default compiler/runtime control separate from WASM configuration."""
    if backend == "pecos":
        data = pc.sim(pc.Guppy(control_without_wasm)).qubits(1).quantum(pc.state_vector()).seed(42).run(3).to_dict()
        shots = [dict(zip(data, values, strict=True)) for values in zip(*data.values(), strict=True)]
    else:
        import selene_sim

        instance = selene_sim.build(control_without_wasm.compile(), build_dir=tmp_path)
        shots = [
            dict(shot)
            for shot in instance.run_shots(
                simulator=selene_sim.Quest(),
                n_qubits=1,
                n_shots=3,
                n_processes=1,
                random_seed=42,
            )
        ]
    assert shots == [{"sum": 5, "product": 20, "measured": 1}] * 3


@pytest.mark.parametrize("wrapper", ["guppy", "hugr"])
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("mixed_arithmetic", {"value": [-4.25] * 4, "integer": [-4] * 4}),
        ("float_feedback", {"measured": [1] * 4}),
        ("float_state", {"initial": [0.0] * 4, "retained": [-2.5] * 4}),
        ("float_loop", {"value": [3.75] * 4, "measured": [1] * 4}),
    ],
)
def test_pecos_float_wasm(examples, wrapper, name, expected):
    pytest.importorskip("pecos_selene_wasm", reason="requires quantum-pecos[wasm]")
    example = examples["floating_point"]
    function = getattr(example, name)
    modules = {str(example.WASM_PATH): example.WASM_PATH.read_bytes()}
    program = (
        pc.Guppy(function, wasm_modules=modules)
        if wrapper == "guppy"
        else pc.Hugr(function.compile().to_bytes(), wasm_modules=modules)
    )
    data = pc.sim(program).qubits(1).quantum(pc.state_vector()).run(4).to_dict()
    assert data == expected


@pytest.mark.parametrize("wrapper", ["guppy", "hugr"])
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("stateful_loop", {"three": 3}),
        ("loop_control", {"zero": 99, "one": 1, "nested": 7, "condition": 10, "steps": 3}),
        ("loop_helpers", {"helper": 3, "after": 4}),
    ],
)
def test_pecos_wasm_loops(examples, wrapper, name, expected):
    pytest.importorskip("pecos_selene_wasm", reason="requires quantum-pecos[wasm]")
    example = examples["stateful"]
    function = getattr(example, name)
    modules = {str(example.WASM_PATH): example.WASM_PATH.read_bytes()}
    program = (
        pc.Guppy(function, wasm_modules=modules)
        if wrapper == "guppy"
        else pc.Hugr(function.compile().to_bytes(), wasm_modules=modules)
    )
    data = pc.sim(program).qubits(1).quantum(pc.state_vector()).workers(2).run(4).to_dict()
    assert data == {label: [value] * 4 for label, value in expected.items()}


@pytest.mark.parametrize("wrapper", ["guppy", "hugr"])
def test_external_loop_uses_safe_frontend_after_normal_cache(examples, monkeypatch, wrapper):
    import re

    from guppylang.library import link_name

    plugin = pytest.importorskip("pecos_selene_wasm", reason="requires quantum-pecos[wasm]")
    example = examples["stateful"]
    counter_module = example.Counter

    @guppy.declare
    @link_name("wasm_test_increment")
    def external(value: int) -> int: ...

    @guppy
    def program() -> None:
        counter = counter_module(nat(0))
        result("counter", counter.increment())
        counter.discard()
        value = 0
        for _ in range(3):
            value = external(value)
        result("total", value)

    # Populate the ordinary per-definition cache first. WASM must not reuse it.
    normal = pc.Guppy(program).hugr_bytes
    modules = {str(example.WASM_PATH): example.WASM_PATH.read_bytes()}
    prepared = pc.Guppy(program, wasm_modules=modules)
    safe_bytes = prepared.hugr_bytes
    assert safe_bytes is prepared.hugr_bytes
    assert pc.Guppy(program).hugr_bytes is normal
    assert safe_bytes != normal
    original_compile = plugin.compile_wasm

    def compile_with_native_fixture(*args, **kwargs):
        ir = original_compile(*args, **kwargs)
        # Implement the declared add-one function independently of lowering.
        body = r"define i64 \1(i64 %n) {\nentry:\n  %next = add i64 %n, 1\n  ret i64 %next\n}"
        ir, count = re.subn(
            r"^declare i64 (@[^\s(]*wasm_test_increment[^\s(]*)\(i64\)[^\n]*$",
            body,
            ir,
            flags=re.MULTILINE,
        )
        assert count == 1
        return ir

    monkeypatch.setattr(plugin, "compile_wasm", compile_with_native_fixture)
    executable = prepared if wrapper == "guppy" else pc.Hugr(safe_bytes, wasm_modules=modules)
    data = pc.sim(executable).qubits(1).quantum(pc.state_vector()).run(2).to_dict()
    assert data == {"counter": [1, 1], "total": [3, 3]}


def test_missing_wasm_extra_has_install_guidance(monkeypatch):
    monkeypatch.setitem(sys.modules, "pecos_selene_wasm", None)
    with pytest.raises(ImportError, match=r"quantum-pecos\[wasm\]"):
        _ = pc.Guppy(control_without_wasm, wasm_modules={}).hugr_bytes


@pytest.mark.parametrize("wrapper", ["guppy", "hugr"])
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("indexed_arithmetic", {"sum": 4, "product": 20, "named": 22}),
        ("indexed_feedback", {"value": 1.5, "measured": 1}),
        ("indexed_state", {"three": 3, "reset": 1}),
    ],
)
def test_pecos_indexed_exports(examples, wrapper, name, expected):
    plugin = pytest.importorskip("pecos_selene_wasm", reason="requires quantum-pecos[wasm]")
    example = examples["indexed_exports"]
    function = getattr(example, name)
    modules = {str(example.WASM_PATH): example.WASM_PATH.read_bytes()}
    program = (
        pc.Guppy(function, wasm_modules=modules)
        if wrapper == "guppy"
        else pc.Hugr(plugin.compile_guppy(function).to_bytes(), wasm_modules=modules)
    )
    data = pc.sim(program).qubits(1).quantum(pc.state_vector()).workers(2).run(4).to_dict()
    assert data == {label: [value] * 4 for label, value in expected.items()}


def test_pecos_indexed_export_error_recovery(examples):
    pytest.importorskip("pecos_selene_wasm", reason="requires quantum-pecos[wasm]")
    example = examples["indexed_exports"]
    bad_modules = {str(example.WASM_PATH): b'(module (func (export "only")))'}
    with pytest.raises(RuntimeError, match=r"out of range|2001"):
        pc.sim(pc.Guppy(example.indexed_feedback, wasm_modules=bad_modules)).qubits(1).quantum(pc.state_vector()).run(1)
    modules = {str(example.WASM_PATH): example.WASM_PATH.read_bytes()}
    data = (
        pc.sim(pc.Guppy(example.indexed_feedback, wasm_modules=modules))
        .qubits(1)
        .quantum(pc.state_vector())
        .run(2)
        .to_dict()
    )
    assert data == {"value": [1.5, 1.5], "measured": [1, 1]}
