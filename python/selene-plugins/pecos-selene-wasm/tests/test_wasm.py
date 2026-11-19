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
    for name in ("arithmetic", "stateful", "floating_point"):
        spec = importlib.util.spec_from_file_location(f"wasm_{name}", EXAMPLES / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


def run(example, function, tmp_path, *, shots=4, processes=1, modules=None, platform="helios", opt_level=2):
    if modules is None:
        modules = {str(example.WASM_PATH): example.WASM_PATH.read_bytes()}
    runner = build(function.compile(), modules, build_dir=tmp_path, platform=platform, opt_level=opt_level)
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


@pytest.mark.parametrize("platform", ["helios", "sol"])
@pytest.mark.parametrize("opt_level", [0, 1, 2, 3])
@pytest.mark.parametrize(
    ("module", "name", "expected"),
    [
        ("stateful", "stateful_loop", {"three": 3}),
        ("stateful", "loop_control", {"zero": 99, "one": 1, "nested": 7, "condition": 10, "steps": 3}),
        ("stateful", "loop_helpers", {"helper": 3, "after": 4}),
        ("floating_point", "float_loop", {"value": 3.75, "measured": 1}),
    ],
)
def test_loop_results_survive_public_compiler(examples, tmp_path, module, name, expected, platform, opt_level):
    example = examples[module]
    assert run(example, getattr(example, name), tmp_path, platform=platform, opt_level=opt_level) == [expected] * 4


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


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("mixed_arithmetic", {"value": -4.25, "integer": -4}),
        ("float_feedback", {"measured": 1}),
        ("float_state", {"initial": 0.0, "retained": -2.5}),
    ],
)
def test_float_signatures_and_feedback(examples, tmp_path, name, expected):
    example = examples["floating_point"]
    assert run(example, getattr(example, name), tmp_path) == [expected] * 4


def test_float_signature_mismatch(examples, tmp_path):
    example = examples["floating_point"]
    modules = {
        str(example.WASM_PATH): b'(module (func (export "affine") (param i64 i64 i64) (result i64) i64.const 0))',
    }
    with pytest.raises(SelenePanicError, match="signature mismatch"):
        run(example, example.float_feedback, tmp_path, modules=modules)


def test_benchmark_checks_results_and_records_timings(tmp_path):
    import json
    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, str(EXAMPLES / "benchmark.py"), "--calls", "0", "2", "--shots", "2", "--repeats", "1"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    report = json.loads(completed.stdout)
    assert report["versions"]["selene-hugr-qis-compiler"]
    assert [record["calls_per_shot"] for record in report["results"]] == [0, 2]
    for record in report["results"]:
        assert record["shots"] == 2
        assert len(record["repeat_seconds"]) == 1
        assert record["backend_prepare_seconds"] >= 0
    assert report["results"][0]["amortized_seconds_per_call"] is None


@pytest.mark.parametrize("opt_level", [0, 2])
def test_existing_external_function_value_and_discarded_result(examples, tmp_path, opt_level):
    import re

    from guppylang import guppy
    from guppylang.library import link_name
    from guppylang.std.builtins import Function, nat, result
    from hugr import ops
    from pecos_selene_wasm import WasmPlugin
    from selene_sim import build as build_llvm

    example = examples["stateful"]
    counter_module = example.Counter

    @guppy.declare
    @link_name("wasm_test_increment")
    def external(value: int) -> int: ...

    @guppy
    def target() -> Function[[int], int]:
        return external

    @guppy
    def program() -> None:
        counter = counter_module(nat(0))
        result("counter", counter.increment())
        counter.discard()
        value = 0
        for _ in range(3):
            function = target()
            value = function(value)
        result("total", value)
        external(100)

    # Preserve first-class function loads before PECOS lowering. Default Guppy
    # optimization can already fold ordinary externs incorrectly at this stage.
    package = program.with_minimal_opt().compile()
    assert any(isinstance(package.modules[0][node].op, ops.LoadFunc) for node in package.modules[0])
    ir = compile_wasm(package, {str(example.WASM_PATH): example.WASM_PATH.read_bytes()}, opt_level=opt_level)
    label = b"USER:INT:external"
    encoded = "".join(f"\\{byte:02X}" for byte in bytes([len(label)]) + label)
    ir += f'\n@fixture_label = private constant [{len(label) + 1} x i8] c"{encoded}"\n'
    # A tiny independent native fixture implements this ordinary declaration.
    # Printing its result also makes the discarded final call observable.
    body = (
        r"define i64 \1(i64 %n) {\nentry:\n  %next = add i64 %n, 1\n"
        f"  call void @print_int(ptr @fixture_label, i64 {len(label)}, i64 %next)\n"
        "  ret i64 %next\n}"
    )
    ir, count = re.subn(
        r"^declare i64 (@[^\s(]*wasm_test_increment[^\s(]*)\(i64\)[^\n]*$",
        body,
        ir,
        flags=re.MULTILINE,
    )
    assert count == 1
    runner = build_llvm(ir, utilities=[WasmPlugin()], opt_level=opt_level, build_dir=tmp_path)
    shots = [dict(shot) for shot in runner.run_shots(simulator=Quest(), n_qubits=1, n_shots=2, n_processes=1)]
    assert shots == [{"counter": 1, "total": 3, "external": 101}] * 2


@pytest.mark.parametrize("opt_level", [0, 2])
def test_hugr_tail_loop_orders_context_helpers(examples, tmp_path, opt_level):
    from hugr import ops, tys
    from hugr.build.function import Module
    from hugr.package import Package
    from hugr.std.int import INT_OPS_EXTENSION, IntVal, int_t

    example = examples["stateful"]
    source = Package.from_bytes(example.increment_context.with_minimal_opt().compile().to_bytes())
    module = Module()
    main = module.define_function("main", [], visibility="Public")
    graph = module.hugr
    mapping = graph.insert_hugr(source.modules[0], graph.module_root)
    helper = mapping[source.modules[0].entrypoint]
    loop = main.add_tail_loop([], [main.load(IntVal(0, 6))])
    # No data dependency orders these leases. The first result is discarded.
    loop.call(helper)
    value = loop.call(helper)
    compare = INT_OPS_EXTENSION.get_op("ige_s").instantiate(
        [tys.BoundedNatArg(6)],
        tys.FunctionType([int_t(6), int_t(6)], [tys.Bool]),
    )
    done = loop.add_op(compare, value, loop.load(IntVal(4, 6)))
    loop.set_loop_outputs(done, value)
    record_graph = example.stateful_loop.compile().modules[0]
    record = next(
        record_graph[node].op
        for node in record_graph
        if isinstance(record_graph[node].op, ops.ExtOp) and record_graph[node].op.op_def().name == "result_int"
    )
    record_node = main.add_op(record, loop.parent_node)
    main.add_state_order(record_node, main.output_node)
    main.set_outputs()
    graph.entrypoint = main.parent_node
    program = Package([graph], source.extensions)
    runner = build(
        program,
        {str(example.WASM_PATH): example.WASM_PATH.read_bytes()},
        opt_level=opt_level,
        build_dir=tmp_path,
    )
    shots = [dict(shot) for shot in runner.run_shots(simulator=Quest(), n_qubits=1, n_shots=4, n_processes=2)]
    assert shots == [{"three": 4}] * 4
