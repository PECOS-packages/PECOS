"""Real public compiler + native utility tests; no proprietary dependencies."""

import importlib.util
import re
import reprlib
import sys
from pathlib import Path

import pytest
from pecos_selene_wasm import build, compile_guppy, compile_wasm
from selene_sim import Quest
from selene_sim.exceptions import SelenePanicError

EXAMPLES = Path(__file__).resolve().parents[4] / "examples/guppy_wasm"


@pytest.fixture(scope="module")
def examples():
    modules = {}
    for name in (
        "arithmetic",
        "stateful",
        "floating_point",
        "indexed_exports",
        "integer32",
        "mixed_integers",
        "memory_lifecycle",
        "managed_objects",
    ):
        spec = importlib.util.spec_from_file_location(f"wasm_{name}", EXAMPLES / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


def run(example, function, tmp_path, *, shots=4, processes=1, modules=None, platform="helios", opt_level=2):
    if modules is None:
        modules = {str(example.WASM_PATH): example.WASM_PATH.read_bytes()}
    runner = build(function, modules, build_dir=tmp_path, platform=platform, opt_level=opt_level)
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
@pytest.mark.parametrize("input_form", ["source", "package", "bytes"])
def test_existing_external_function_value_and_discarded_result(examples, tmp_path, opt_level, input_form):
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
    package = compile_guppy(program)
    assert any(isinstance(package.modules[0][node].op, ops.LoadFunc) for node in package.modules[0])
    prepared = {"source": program, "package": package, "bytes": package.to_bytes()}[input_form]
    ir = compile_wasm(prepared, {str(example.WASM_PATH): example.WASM_PATH.read_bytes()}, opt_level=opt_level)
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


@pytest.mark.parametrize("serialized", [False, True])
def test_unprepared_external_loop_is_rejected(examples, serialized):
    from guppylang import guppy
    from guppylang.library import link_name
    from guppylang.std.builtins import nat, result

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

    package = program.compile()
    with pytest.raises(ValueError, match="safe Guppy frontend"):
        compile_wasm(
            package.to_bytes() if serialized else package,
            {str(example.WASM_PATH): example.WASM_PATH.read_bytes()},
        )


@pytest.mark.parametrize("platform", ["helios", "sol"])
@pytest.mark.parametrize("opt_level", [0, 2])
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("indexed_arithmetic", {"sum": 4, "product": 20, "named": 22}),
        ("indexed_feedback", {"value": 1.5, "measured": 1}),
        ("indexed_state", {"three": 3, "reset": 1}),
    ],
)
def test_indexed_exports(examples, tmp_path, platform, opt_level, name, expected):
    example = examples["indexed_exports"]
    assert (
        run(
            example,
            getattr(example, name),
            tmp_path,
            platform=platform,
            opt_level=opt_level,
            processes=2,
        )
        == [expected] * 4
    )


@pytest.mark.parametrize("binary", [False, True])
def test_index_uses_export_order_from_supplied_bytes(examples, tmp_path, binary):
    from wasmtime import wat2wasm

    example = examples["indexed_exports"]
    # Names in the declaration file are not bindings. Replacing the explicitly
    # supplied export at position 3 changes execution even when its name changes.
    wat = example.WASM_PATH.read_text().replace('(export "affine"', '(export "renamed"').replace("f64.add", "f64.sub")
    data = bytes(wat2wasm(wat)) if binary else wat.encode()
    assert (
        run(example, example.indexed_feedback, tmp_path, modules={str(example.WASM_PATH): data})
        == [
            {"value": -6.5, "measured": 0},
        ]
        * 4
    )


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b'(module (func (export "only")))', "out of range"),
        (b'(module (func (export "only")) (memory (export "memory") 1))', "not a function"),
        (
            b'(module (memory (export "memory") 1) (func (export "wrong") (param f64 f64) (result f64) local.get 0))',
            "signature mismatch",
        ),
    ],
)
def test_invalid_indexed_export_traps(examples, tmp_path, data, message):
    example = examples["indexed_exports"]
    with pytest.raises(SelenePanicError, match=message):
        run(example, example.indexed_arithmetic, tmp_path, modules={str(example.WASM_PATH): data})


@pytest.mark.parametrize("index", [-1, 2**64, 2**64 + 2])
def test_index_must_fit_unsigned_64_bits(examples, index):
    from hugr import ops, tys
    from hugr.envelope import EnvelopeConfig, EnvelopeFormat

    example = examples["indexed_exports"]
    package = compile_guppy(example.indexed_arithmetic)
    for graph in package.modules:
        for node in graph:
            operation = graph[node].op
            if isinstance(operation, ops.ExtOp) and operation.op_def().name == "lookup_by_id":
                operation.args[0] = tys.BoundedNatArg(index)
    # The JSON envelope permits out-of-range BoundedNat values; binary model
    # encoding rejects them earlier. They must never reach LLVM's truncation.
    envelope = package.to_bytes(EnvelopeConfig(format=EnvelopeFormat.JSON))
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        compile_wasm(envelope, {str(example.WASM_PATH): example.WASM_PATH.read_bytes()})


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "boundaries",
            {
                "sum": 5,
                "overflow": -2147483648,
                "underflow": 2147483647,
                "wide": 5,
                "negative_wide": 2147483647,
                "indexed": -5,
            },
        ),
        ("state_and_feedback", {"initial": 0, "retained": -7, "measured": 1}),
    ],
)
def test_i32_guppy_exports(examples, tmp_path, name, expected):
    example = examples["integer32"]
    runner = build(
        getattr(example, name),
        {str(example.WASM_PATH): example.WASM_PATH.read_bytes()},
        build_dir=tmp_path,
        wasm_int_width=32,
    )
    for _ in range(2):
        assert [
            dict(shot)
            for shot in runner.run_shots(
                simulator=Quest(),
                n_qubits=1,
                n_shots=4,
                n_processes=2,
                random_seed=42,
            )
        ] == [expected] * 4


def test_i32_mode_rejects_i64_module(examples, tmp_path):
    example = examples["arithmetic"]
    runner = build(
        example.addition,
        {str(example.WASM_PATH): example.WASM_PATH.read_bytes()},
        build_dir=tmp_path,
        wasm_int_width=32,
    )
    with pytest.raises(SelenePanicError, match="signature mismatch"):
        [dict(shot) for shot in runner.run_shots(simulator=Quest(), n_qubits=1, n_shots=1)]


@pytest.mark.parametrize("width", [16, 32.0, True, "32", "AUTO"])
def test_invalid_integer_width_is_rejected(examples, width):
    example = examples["integer32"]
    with pytest.raises(ValueError, match=re.escape(f"wasm_int_width must be 32 or 64, or 'auto'; got {width!r}")):
        compile_wasm(example.boundaries, {str(example.WASM_PATH): example.WASM_PATH.read_bytes()}, wasm_int_width=width)


@pytest.mark.parametrize("timeout", [True, "2", None, 0, -1, float("nan"), float("inf"), 10**400])
def test_invalid_timeout_is_rejected(timeout):
    message = f"wasm_timeout_seconds must be a positive finite number of seconds; got {reprlib.repr(timeout)}"
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        compile_wasm(b"", {}, wasm_timeout_seconds=timeout)


@pytest.mark.parametrize(
    "timeout",
    [2.5, float.fromhex("0x0.0000000000001p-1022"), float.fromhex("0x1.fffffffffffffp1023")],
)
def test_custom_timeout_ir_and_execution(examples, tmp_path, timeout):
    import struct

    example = examples["arithmetic"]
    modules = {str(example.WASM_PATH): example.WASM_PATH.read_bytes()}
    llvm = compile_wasm(example.addition, modules, wasm_timeout_seconds=timeout)
    assert "call i32 @pecos_wasm_v2_register" in llvm
    assert "double 0x" + struct.pack(">d", timeout).hex().upper() in llvm
    # The smallest positive limit is useful for checking encoding/tick conversion,
    # but wall-clock scheduling can exhaust its single tick before execution.
    if timeout > 1:
        runner = build(example.addition, modules, build_dir=tmp_path, wasm_timeout_seconds=timeout)
        assert [dict(s) for s in runner.run_shots(simulator=Quest(), n_qubits=1, n_shots=2)] == [{"sum": 5}] * 2


def test_default_timeout_preserves_v1_registration(examples):
    example = examples["arithmetic"]
    modules = {str(example.WASM_PATH): example.WASM_PATH.read_bytes()}
    for options in ({}, {"wasm_timeout_seconds": 1.0}):
        llvm = compile_wasm(example.addition, modules, **options)
        assert "call i32 @pecos_wasm_v1_register" in llvm
        assert "pecos_wasm_v2_register" not in llvm
        assert "pecos_wasm_v3_register" not in llvm


def test_custom_timeout_requires_new_native_runtime(examples, tmp_path, monkeypatch):
    import pecos_selene_wasm

    monkeypatch.setattr(pecos_selene_wasm.ctypes, "CDLL", lambda _path: object())
    example = examples["arithmetic"]
    with pytest.raises(RuntimeError, match="rebuild or upgrade"):
        build(
            example.addition,
            {str(example.WASM_PATH): example.WASM_PATH.read_bytes()},
            build_dir=tmp_path,
            wasm_timeout_seconds=2.0,
        )


@pytest.mark.parametrize("value", [0, 1, None, "false"])
def test_invalid_auto_init_is_rejected(value):
    with pytest.raises(ValueError, match=rf"^wasm_auto_init must be True or False; got {re.escape(repr(value))}$"):
        compile_wasm(b"", {}, wasm_auto_init=value)


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
        ("wasm_auto_init", "wasm_auto_init must be True or False; got "),
        ("wasm_timeout_seconds", "wasm_timeout_seconds must be a positive finite number of seconds; got "),
        ("wasm_int_width", "wasm_int_width must be 32 or 64, or 'auto'; got "),
    ],
)
def test_rejected_option_errors_survive_unprintable_values(value, described, option, prefix):
    with pytest.raises(ValueError, match=f"^{re.escape(prefix)}{described}$"):
        compile_wasm(b"", {}, **{option: value})


@pytest.mark.parametrize("auto_init", [True, False])
@pytest.mark.parametrize("explicit", [True, False])
def test_memory_lifecycle_and_initialization_policy(examples, tmp_path, auto_init, explicit):
    example = examples["memory_lifecycle"]
    modules = {str(example.WASM_PATH): example.WASM_PATH.read_bytes()}
    program = example.explicit_initialization if explicit else example.memory_operations
    initial = 42 if auto_init else 11
    expected = (
        {"before": initial, "after": initial + 31}
        if explicit
        else {
            "initialized": initial,
            "incremented": initial + 1,
            "old_pages": 1,
            "pages": 2,
            "zero": 0,
            "stored": -7,
            "indirect": 12,
            "growth_refused": -1,
        }
    )
    llvm = compile_wasm(program, modules, wasm_auto_init=auto_init, wasm_timeout_seconds=2.5)
    assert ("call i32 @pecos_wasm_v3_register" in llvm) is (not auto_init)
    runner = build(program, modules, build_dir=tmp_path, wasm_auto_init=auto_init, wasm_timeout_seconds=2.5)
    for _ in range(2):
        assert [dict(s) for s in runner.run_shots(simulator=Quest(), n_qubits=1, n_shots=4, n_processes=2)] == [
            expected,
        ] * 4


def test_auto_init_requires_new_native_runtime(examples, tmp_path, monkeypatch):
    import pecos_selene_wasm

    monkeypatch.setattr(pecos_selene_wasm.ctypes, "CDLL", lambda _path: object())
    example = examples["arithmetic"]
    with pytest.raises(RuntimeError, match="rebuild or upgrade"):
        build(
            example.addition,
            {str(example.WASM_PATH): example.WASM_PATH.read_bytes()},
            build_dir=tmp_path,
            wasm_auto_init=False,
        )


def test_initialization_policy_changes_in_reused_build_directory(examples, tmp_path):
    example = examples["memory_lifecycle"]
    modules = {str(example.WASM_PATH): example.WASM_PATH.read_bytes()}
    for auto_init, expected in ((True, 42), (False, 11), (True, 42)):
        runner = build(example.memory_operations, modules, build_dir=tmp_path, wasm_auto_init=auto_init)
        shots = [dict(s) for s in runner.run_shots(simulator=Quest(), n_qubits=1, n_shots=1)]
        assert shots[0]["initialized"] == expected


def test_managed_objects_use_scalar_abi_and_fresh_stores(examples, tmp_path):
    example = examples["managed_objects"]
    expected = {"initial": 7, "next": 8, "sum": 5, "retained": 8, "independent": 7}
    runner = build(
        example.managed_state,
        {str(example.WASM_PATH): example.RUNTIME_WASM_PATH.read_bytes()},
        build_dir=tmp_path,
    )
    for _ in range(2):
        shots = [dict(s) for s in runner.run_shots(simulator=Quest(), n_qubits=1, n_shots=4, n_processes=2)]
        assert shots == [expected] * 4


@pytest.mark.parametrize("backend", ["selene", "pecos"])
def test_gc_allocating_loop_times_out_and_recovers(tmp_path, backend):
    if backend == "pecos":
        pytest.importorskip("pecos")
    import subprocess
    import sys

    script = tmp_path / "gc_timeout.py"
    script.write_text(
        """
import importlib.util, sys
from pathlib import Path
from tempfile import TemporaryDirectory
spec = importlib.util.spec_from_file_location("arithmetic", Path(sys.argv[2]) / "arithmetic.py")
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)
loop = b"""
        + repr("""(module
(type $cell (struct (field i64)))
(global $last (mut (ref null $cell)) (ref.null $cell))
(func (export "add") (param i64 i64) (result i64)
  (loop $again local.get 0 struct.new $cell global.set $last br $again)
  i64.const 0))""")
        + """
def execute(data, directory, timeout=2.0):
    modules = {str(example.WASM_PATH): data}
    if sys.argv[1] == "selene":
        from pecos_selene_wasm import build
        from selene_sim import Quest
        runner = build(example.addition, modules, build_dir=Path(directory), wasm_timeout_seconds=timeout)
        return [dict(s) for s in runner.run_shots(simulator=Quest(), n_qubits=1, n_shots=1)]
    import pecos as pc
    program = pc.Guppy(example.addition, wasm_modules=modules, wasm_timeout_seconds=timeout)
    return pc.sim(program).qubits(1).quantum(pc.state_vector()).run(1).to_dict()
with TemporaryDirectory() as directory:
    try:
        execute(loop, directory, timeout=0.05)
    except Exception as error:
        if "timed out" not in str(error):
            raise
    else:
        raise RuntimeError("allocating loop did not time out")
with TemporaryDirectory() as directory:
    result = execute(example.WASM_PATH.read_bytes(), directory)
    if result not in ([{"sum": 5}], {"sum": [5]}):
        raise RuntimeError("execution did not recover after GC timeout")
""",
    )
    completed = subprocess.run(
        [sys.executable, str(script), backend, str(EXAMPLES)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("width", ["auto", 32, 64])
def test_mixed_integer_widths(examples, tmp_path, width):
    example = examples["mixed_integers"]
    modules = {str(example.WASM_PATH): example.WASM_PATH.read_bytes()}
    llvm = compile_wasm(example.mixed, modules, wasm_int_width=width)
    assert ("@pecos_wasm_v3_lookup" in llvm) == (width == "auto")
    assert "call i32 @pecos_wasm_v3_register" not in llvm
    runner = build(example.mixed, modules, build_dir=tmp_path, wasm_int_width=width)

    def shots():
        return [dict(shot) for shot in runner.run_shots(simulator=Quest(), n_qubits=1, n_shots=4, n_processes=2)]

    if width == "auto":
        expected = {"small": -2147483648, "large": 4294967308, "small_indexed": -3, "large_indexed": 4294967293}
        assert shots() == [expected] * 4
        assert shots() == [expected] * 4
    else:
        with pytest.raises(SelenePanicError, match="signature mismatch"):
            shots()


@pytest.mark.parametrize("missing", ["pecos_wasm_v3_lookup", "pecos_wasm_v3_lookup_by_id"])
def test_automatic_width_requires_new_native_lookup(examples, tmp_path, monkeypatch, missing):
    from types import SimpleNamespace

    import pecos_selene_wasm

    symbols = {name: object() for name in ("pecos_wasm_v3_lookup", "pecos_wasm_v3_lookup_by_id") if name != missing}
    monkeypatch.setattr(pecos_selene_wasm.ctypes, "CDLL", lambda _path: SimpleNamespace(**symbols))
    example = examples["mixed_integers"]
    with pytest.raises(RuntimeError, match="Automatic WASM integer widths require"):
        build(
            example.mixed,
            {str(example.WASM_PATH): example.WASM_PATH.read_bytes()},
            build_dir=tmp_path,
            wasm_int_width="auto",
        )


def test_automatic_width_preserves_float_lanes(examples, tmp_path):
    example = examples["floating_point"]
    runner = build(
        example.mixed_arithmetic,
        {str(example.WASM_PATH): example.WASM_PATH.read_bytes()},
        build_dir=tmp_path,
        wasm_int_width="auto",
    )
    assert [dict(shot) for shot in runner.run_shots(simulator=Quest(), n_qubits=1, n_shots=2)] == [
        {"value": -4.25, "integer": -4},
    ] * 2
