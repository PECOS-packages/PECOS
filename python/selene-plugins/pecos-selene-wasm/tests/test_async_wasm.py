"""Public async execution, typed results, worker state, and bounded failure cleanup."""

import importlib.util
import re
import reprlib
import subprocess
import sys
from pathlib import Path

import pytest
from pecos_selene_wasm import build, compile_guppy, compile_wasm
from selene_sim import Quest

EXAMPLES = Path(__file__).resolve().parents[4] / "examples/guppy_wasm"


@pytest.fixture(scope="module")
def examples():
    loaded = {}
    for name in ("async_calls", "arithmetic", "worker_state", "mixed_integers", "floating_point", "managed_objects"):
        spec = importlib.util.spec_from_file_location(f"async_{name}", EXAMPLES / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded


def prepare(backend, program, modules, directory, *, execution_workers=1, **options):
    if backend == "selene":
        runner = build(program, modules, build_dir=directory, **options)

        def run(shots):
            return [
                dict(shot)
                for shot in runner.run_shots(
                    simulator=Quest(),
                    n_qubits=1,
                    n_shots=shots,
                    n_processes=execution_workers,
                )
            ]

    else:
        pc = pytest.importorskip("pecos")
        constructor = pc.Hugr.from_bytes if isinstance(program, bytes) else pc.Guppy
        if hasattr(program, "modules"):
            program = program.to_bytes()
            constructor = pc.Hugr.from_bytes
        runner = (
            pc.sim(constructor(program, wasm_modules=modules, **options))
            .qubits(1)
            .quantum(pc.state_vector())
            .workers(execution_workers)
            .build()
        )

        def run(shots):
            output = runner.run(shots).to_dict()
            return [{name: values[i] for name, values in output.items()} for i in range(shots)]

    return run


@pytest.mark.parametrize("backend", ["selene", "pecos"])
@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("execution_workers", [1, 3])
@pytest.mark.parametrize("feedback", [False, True])
def test_deferred_calls_and_precompiled_bytes(examples, tmp_path, backend, workers, execution_workers, feedback):
    example = examples["async_calls"]
    program = example.deferred_program(example.quantum_feedback if feedback else example.independent_calls).to_bytes()
    run = prepare(
        backend,
        program,
        {str(example.WASM_PATH): example.WASM_PATH.read_bytes()},
        tmp_path,
        execution_workers=execution_workers,
        wasm_async_workers=workers,
    )
    expected = {"first": example.expected_value(1000000), "second": example.expected_value(2000000)}
    if feedback:
        expected.update(second=example.expected_value(2000001), measured=True)
    for _ in range(2):
        assert run(5) == [expected] * 5


@pytest.mark.parametrize("backend", ["selene", "pecos"])
@pytest.mark.parametrize("scope", ["shot", "worker"])
@pytest.mark.parametrize("execution_workers", [1, 3])
@pytest.mark.parametrize("shots", [1, 5])
def test_async_worker_state_and_fresh_batches(examples, tmp_path, backend, scope, execution_workers, shots):
    example = examples["worker_state"]
    run = prepare(
        backend,
        example.counter,
        {str(example.WASM_PATH): example.WASM_PATH.read_bytes()},
        tmp_path,
        execution_workers=execution_workers,
        wasm_async_workers=2,
        wasm_state_scope=scope,
    )
    expected = (
        [1] * shots
        if scope == "shot"
        else sorted(
            value
            for worker in range(execution_workers)
            for value in range(1, shots // execution_workers + (worker < shots % execution_workers) + 1)
        )
    )
    for _ in range(2):
        assert sorted(shot["count"] for shot in run(shots)) == expected


@pytest.mark.parametrize("backend", ["selene", "pecos"])
@pytest.mark.parametrize(
    ("name", "function", "expected", "width"),
    [
        ("floating_point", "mixed_arithmetic", {"value": -4.25, "integer": -4}, 64),
        (
            "mixed_integers",
            "mixed",
            {"small": -2147483648, "small_indexed": -3, "large": 4294967308, "large_indexed": 4294967293},
            "auto",
        ),
        ("managed_objects", "managed_state", {"initial": 7, "next": 8, "sum": 5, "retained": 8, "independent": 7}, 64),
    ],
)
def test_async_scalar_and_managed_state(examples, tmp_path, backend, name, function, expected, width):
    example = examples[name]
    payload = getattr(example, "RUNTIME_WASM_PATH", example.WASM_PATH).read_bytes()
    run = prepare(
        backend,
        getattr(example, function),
        {str(example.WASM_PATH): payload},
        tmp_path,
        wasm_async_workers=2,
        wasm_int_width=width,
    )
    assert run(3) == [expected] * 3


@pytest.mark.parametrize("workers", [-1, 65, True, 2.0, "2", None])
def test_invalid_async_worker_count(workers):
    message = f"wasm_async_workers must be an integer between 0 and 64; got {reprlib.repr(workers)}"
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        compile_wasm(b"", {}, wasm_async_workers=workers)


@pytest.mark.parametrize("workers", [-1, 65, True, 2.0, "2", None])
def test_pecos_rejects_invalid_async_worker_count(workers):
    pc = pytest.importorskip("pecos")
    message = f"wasm_async_workers must be an integer between 0 and 64; got {reprlib.repr(workers)}"
    for constructor in (pc.Guppy, pc.Hugr.from_bytes):
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            constructor(b"", wasm_modules={}, wasm_async_workers=workers)


def test_pecos_async_workers_require_explicit_modules():
    pc = pytest.importorskip("pecos")
    message = "wasm_async_workers=2 requires explicit wasm_modules; pass wasm_modules={module_name: wasm_bytes}"
    for constructor in (pc.Guppy, pc.Hugr.from_bytes):
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            constructor(b"", wasm_async_workers=2)


def test_async_symbol_is_opt_in_and_preflighted(examples, tmp_path, monkeypatch):
    import pecos_selene_wasm

    example = examples["arithmetic"]
    modules = {str(example.WASM_PATH): example.WASM_PATH.read_bytes()}
    default = compile_wasm(example.addition, modules)
    assert "pecos_wasm_v2_call" not in default
    enabled = compile_wasm(example.addition, modules, wasm_async_workers=2)
    assert "call i32 @pecos_wasm_v2_call" in enabled
    monkeypatch.setattr(pecos_selene_wasm.ctypes, "CDLL", lambda _path: object())
    with pytest.raises(RuntimeError, match="Asynchronous WASM calls require"):
        build(example.addition, modules, build_dir=tmp_path, wasm_async_workers=2)


@pytest.mark.parametrize("backend", ["selene", "pecos"])
@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("failure", ["trap", "timeout"])
def test_outstanding_calls_failure_and_recovery_have_watchdog(tmp_path, backend, workers, failure):
    if backend == "pecos":
        pytest.importorskip("pecos")
    script = f"""
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

helper = load({str(Path(__file__).resolve())!r}, "async_test_helpers")
example = load({str(EXAMPLES / "async_calls.py")!r}, "async_probe")
healthy = load({str(EXAMPLES / "arithmetic.py")!r}, "healthy_probe")
trap = "local.get 0 i64.const 1000000 i64.eq if unreachable end" if {failure!r} == "trap" else ""
payload = ('(module (func (export "work") (param i64) (result i64) '
           + trap + " (loop br 0) i64.const 0))").encode()
with TemporaryDirectory() as directory:
    run = helper.prepare({backend!r}, example.deferred_program().to_bytes(),
        {{str(example.WASM_PATH): payload}}, Path(directory) / "bad", wasm_async_workers={workers},
        wasm_timeout_seconds=0.05)
    try:
        run(1)
    except Exception as error:
        expected = "unreachable" if {failure!r} == "trap" else "timed out"
        assert expected in str(error), str(error)
    else:
        raise AssertionError("Expected WASM failure")
    run = helper.prepare({backend!r}, healthy.addition,
        {{str(healthy.WASM_PATH): healthy.WASM_PATH.read_bytes()}}, Path(directory) / "good",
        wasm_async_workers={workers})
    assert run(2) == [{{"sum": 5}}] * 2
"""
    path = tmp_path / "probe.py"
    path.write_text(script)
    completed = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, timeout=60, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
