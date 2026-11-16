"""Real public compiler + native utility tests; no proprietary dependencies."""

import importlib.util
from pathlib import Path

import pytest
from pecos_selene_wasm import build, compile_wasm
from selene_sim import Quest
from selene_sim.exceptions import SelenePanicError

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


def run(example, function, tmp_path, *, shots=4, processes=1, modules=None, platform="helios"):
    if modules is None:
        modules = {str(example.WASM_PATH): example.WASM_PATH.read_bytes()}
    runner = build(function.compile(), modules, build_dir=tmp_path, platform=platform)
    return [
        dict(shot)
        for shot in runner.run_shots(
            simulator=Quest(),
            n_qubits=1,
            n_shots=shots,
            n_processes=processes,
            random_seed=42,
        )
    ]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("addition", {"sum": 5}),
        ("chained_arithmetic", {"product": 20}),
        ("conditional_x", {"measured": 1}),
    ],
)
def test_selene_examples(examples, tmp_path, name, expected):
    example = examples["arithmetic"]
    assert run(example, getattr(example, name), tmp_path) == [expected] * 4


@pytest.mark.parametrize("processes", [1, 2])
def test_state_shot_context_and_process_isolation(examples, tmp_path, processes):
    example = examples["stateful"]
    assert (
        run(example, example.stateful_calls, tmp_path, shots=6, processes=processes)
        == [
            {"three": 3, "four": 4, "independent": 1, "cleared": 1},
        ]
        * 6
    )


def test_trap_propagates(examples, tmp_path):
    example = examples["stateful"]
    with pytest.raises(SelenePanicError, match=r"WASM|wasm|unreachable"):
        run(example, example.trapped, tmp_path)


def test_context_order_across_helpers_and_branches(examples, tmp_path):
    example = examples["stateful"]
    assert run(example, example.context_helpers, tmp_path) == [{"second": 2, "third": 3}] * 4


def test_dead_wasm_branch_can_be_optimized_away(examples, tmp_path):
    example = examples["arithmetic"]
    assert run(example, example.conditional_calls, tmp_path) == [{"value": 6}] * 4


def test_loops_fail_before_public_compiler_can_miscompile(examples):
    example = examples["stateful"]
    with pytest.raises(ValueError, match="loop-carried external-call results"):
        compile_wasm(example.stateful_loop.compile(), {str(example.WASM_PATH): example.WASM_PATH.read_bytes()})


def test_sol_platform_reaches_compiler_and_linker(examples, tmp_path):
    example = examples["arithmetic"]
    assert run(example, example.conditional_x, tmp_path, platform="sol") == [{"measured": 1}] * 4


def test_false_quantum_branch(examples, tmp_path):
    example = examples["arithmetic"]
    modules = {str(example.WASM_PATH): b'(module (func (export "add") (param i64 i64) (result i64) i64.const 0))'}
    assert run(example, example.conditional_x, tmp_path, modules=modules) == [{"measured": 0}] * 4


def test_binary_module_bytes(examples, tmp_path):
    from wasmtime import wat2wasm

    example = examples["arithmetic"]
    modules = {str(example.WASM_PATH): bytes(wat2wasm(example.WASM_PATH.read_text()))}
    assert run(example, example.addition, tmp_path, modules=modules) == [{"sum": 5}] * 4


def test_module_resolution_requires_explicit_bytes(examples):
    example = examples["arithmetic"]
    program = example.addition.compile()
    with pytest.raises(ValueError, match="Supply bytes"):
        compile_wasm(program, {})
    with pytest.raises(TypeError, match="explicit bytes"):
        compile_wasm(program, {str(example.WASM_PATH): example.WASM_PATH})


def test_supplied_bytes_determine_execution(examples, tmp_path):
    example = examples["arithmetic"]
    # The original file remains addition, but explicit bytes change its behavior.
    modules = {str(example.WASM_PATH): b'(module (func (export "add") (param i64 i64) (result i64) i64.const -17))'}
    assert run(example, example.addition, tmp_path, modules=modules) == [{"sum": -17}] * 4


def test_runtime_checks_actual_export_signature(examples, tmp_path):
    example = examples["arithmetic"]
    modules = {str(example.WASM_PATH): b'(module (func (export "add") (param i32 i32) (result i32) i32.const 0))'}
    with pytest.raises(SelenePanicError, match="signature mismatch"):
        run(example, example.addition, tmp_path, modules=modules)


@pytest.mark.parametrize("opt_level", [0, 1, 2, 3])
def test_lowered_llvm_verifies(examples, opt_level):
    import llvmlite.binding as llvm

    example = examples["arithmetic"]
    ir = compile_wasm(
        example.addition.compile(),
        {str(example.WASM_PATH): example.WASM_PATH.read_bytes()},
        opt_level=opt_level,
    )
    llvm.parse_assembly(ir).verify()
