"""Guppy WASM compilation and real execution through the PECOS sim() stack."""

import importlib.util
from pathlib import Path

import pecos as pc
import pytest
from hugr import ops
from hugr.package import Package

EXAMPLES = Path(__file__).resolve().parents[4] / "examples/guppy_wasm"


@pytest.fixture(scope="module")
def examples():
    modules = {}
    for name in ("arithmetic", "stateful"):
        spec = importlib.util.spec_from_file_location(f"wasm_{name}", EXAMPLES / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


@pytest.mark.parametrize(
    ("name", "arguments", "expected"),
    [
        ("add", [2, 3], 5),
        ("add", [-7, 3], -4),
        ("add", [2**40, 3], 2**40 + 3),
        ("multiply", [5, 4], 20),
        ("multiply", [-7, 3], -21),
        ("multiply", [0, 9], 0),
    ],
)
def test_direct_wasm(examples, name, arguments, expected):
    module = pc.WasmForeignObject.from_file(examples["arithmetic"].WASM_PATH)
    module.init()
    try:
        assert module.exec(name, arguments) == expected
    finally:
        module.teardown()


def test_guppy_preserves_wasm_calls(examples):
    package = Package.from_bytes(pc.Guppy(examples["arithmetic"].addition).hugr_bytes)
    names = [
        op.op_def().name
        for graph in package.modules
        for node in graph
        if isinstance(op := graph[node].op, ops.ExtOp) and op.op_def().get_extension().name == "tket.wasm"
    ]
    assert sorted(names) == sorted(["get_context", "lookup_by_name", "call", "read_result", "dispose_context"])


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
