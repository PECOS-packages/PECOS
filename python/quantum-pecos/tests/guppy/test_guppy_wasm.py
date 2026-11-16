"""Guppy WASM declarations, compilation, arithmetic, and runtime support probes."""

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

EXAMPLE_PATH = Path(__file__).resolve().parents[4] / "examples/guppy_wasm/arithmetic.py"


@pytest.fixture(scope="module")
def example():
    """Import the actual example so tests exercise the documented programs."""
    spec = importlib.util.spec_from_file_location("guppy_wasm_arithmetic", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def arithmetic(example):
    module = pc.WasmForeignObject.from_file(example.WASM_PATH)
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
def test_guppy_wasm_hugr_roundtrip(example, name, exports) -> None:
    """PECOS's Guppy wrapper must preserve actual WASM calls and their context."""
    program = pc.Guppy(getattr(example, name))
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


def test_missing_export_is_rejected(example) -> None:
    with pytest.raises(GuppyError, match="isn't exported"):

        @wasm_module(str(example.WASM_PATH))
        class MissingExport:
            @wasm
            def subtract(self, a: int, b: int) -> int: ...


def test_wrong_signature_is_rejected(example) -> None:
    @wasm_module(str(example.WASM_PATH))
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


def test_example_runs_from_another_directory(tmp_path) -> None:
    completed = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert "Compiled conditional_x to HUGR" in completed.stdout
    assert "Direct WASM execution: 2 + 3 = 5; (2 + 3) * 4 = 20" in completed.stdout


@guppy
def control_without_wasm() -> None:
    """Verify both runners and result parsing independently of WASM support."""
    result("sum", 5)
    result("product", 20)
    q = qubit()
    x(q)
    result("measured", measure(q).read())


@pytest.mark.parametrize("backend", ["pecos", "selene"])
def test_runtime_probe_control(example, backend) -> None:
    data = getattr(example, f"run_{backend}")(control_without_wasm, shots=3)
    assert data["sum"] == [5] * 3
    assert data["product"] == [20] * 3
    assert data["measured"] == [1] * 3


@pytest.mark.parametrize("backend", ["pecos", "selene"])
@pytest.mark.parametrize(
    ("name", "label", "expected"),
    [("addition", "sum", 5), ("chained_arithmetic", "product", 20), ("conditional_x", "measured", 1)],
)
def test_guppy_wasm_execution(example, backend, name, label, expected) -> None:
    """Automatically exercise results once the compiler gains WASM support.

    Only the observed missing WASM type is an expected failure. Other build or
    runtime errors, and wrong answers, must fail normally.
    """
    try:
        data = getattr(example, f"run_{backend}")(getattr(example, name), shots=3)
    except RuntimeError as error:
        if "Failed to emit LLVM for function" in str(error) and "Unknown type: module" in str(error):
            pytest.xfail("The default Selene HUGR/QIS compiler path rejects the tket.wasm module type")
        raise
    assert data[label] == [expected] * 3
