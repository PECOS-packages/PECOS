"""Typed named results and program errors on the QIS route."""

import pecos
import pytest
from guppylang import guppy
from guppylang.std.builtins import array, nat, result
from guppylang.std.platform import exit as guppy_exit
from guppylang.std.platform import panic
from guppylang.std.quantum import h, measure, qubit, x
from pecos import Guppy, Qis, sim


@guppy
def typed_results() -> None:
    m = measure(qubit()).read()
    result("i", 7)
    result("f", 2.5)
    result("b", m)
    result("ia", array(3, 9))
    result("fa", array(1.5, 2.5))
    result("u", nat(11))
    result("ua", array(nat(3), nat(9)))
    result("ba", array(m, True))
    result("repeated", 7)
    result("repeated", 9)
    result("single", array(3))
    result("joined", 3)
    result("joined", array(9, 11))
    result("det", False)
    result("det", 0)
    result("det", array(0, 1))
    result("unsigned_det", True)
    result("unsigned_det", nat(0))


def test_typed_named_results() -> None:
    """Scalars, arrays, and repeated tags retain their values and shapes."""
    data = sim(Guppy(typed_results)).classical(pecos.selene_engine()).qubits(1).run(5).to_dict()
    assert data == {
        "i": [7] * 5,
        "f": [2.5] * 5,
        "b": [0] * 5,
        "ia": [[3, 9]] * 5,
        "fa": [[1.5, 2.5]] * 5,
        "u": [11] * 5,
        "ua": [[3, 9]] * 5,
        "ba": [[0, 1]] * 5,
        "repeated": [[7, 9]] * 5,
        "single": [3] * 5,
        "joined": [[3, 9, 11]] * 5,
        "det": [[0, 0, 0, 1]] * 5,
        "unsigned_det": [[1, 0]] * 5,
    }


@pytest.mark.parametrize("seed", [1, 7, 42, 99])
def test_measurement_dependent_integer(seed: int) -> None:
    """Mixed 1/7 outcomes form a uniform integer column across shots and seeds.

    Conversion to ShotMap inside to_dict rejects any per-shot storage type flip;
    the Rust export tests additionally pin the declared type as I64.
    """

    @guppy
    def conditional_result() -> None:
        q = qubit()
        h(q)
        m = measure(q).read()
        count = 1 if m else 7
        result("count", count)
        result("m", m)
        result("constant", 2.5)

    data = sim(Guppy(conditional_result)).classical(pecos.selene_engine()).qubits(1).seed(seed).run(32).to_dict()
    assert set(data) == {"count", "m", "constant"}
    assert set(data["m"]) == {0, 1}
    assert data["count"] == [1 if m else 7 for m in data["m"]]
    assert data["constant"] == [2.5] * 32


def test_integer_aggregation_and_bool_widening() -> None:
    """Integer values and bool widening keep one declared integer type per tag."""

    @guppy
    def aggregate() -> None:
        m = measure(qubit()).read()
        for i in range(4):
            result("c", i)
        result("t", array(0, 1))
        result("t", array(0, 1, 2))
        result("widened", m)
        result("widened", 0)
        result("integer_first", 7)
        result("integer_first", True)

    data = sim(Guppy(aggregate)).classical(pecos.selene_engine()).qubits(1).run(5).to_dict()
    assert data == {
        "c": [[0, 1, 2, 3]] * 5,
        "t": [[0, 1, 0, 1, 2]] * 5,
        "widened": [[0, 0]] * 5,
        "integer_first": [[7, 1]] * 5,
    }


def test_division_by_zero_survives() -> None:
    """A program panic raises with its message and leaves the process usable."""

    @guppy
    def divide_by_zero() -> None:
        result("before", 7)
        q = qubit()
        x(q)
        divisor = 1 - int(measure(q).read())
        result("quotient", 5 // divisor)

    builder = sim(Guppy(divide_by_zero)).classical(pecos.selene_engine()).qubits(1)
    with pytest.raises(RuntimeError, match="Attempted division by 0"):
        builder.run(1)

    following = sim(Guppy(typed_results)).classical(pecos.selene_engine()).qubits(1)
    assert isinstance(following, type(builder))
    assert following.run(2).to_dict()["i"] == [7, 7]


def test_panic_during_operation_collection() -> None:
    """Eager operation collection retains the program's message too."""

    @guppy
    def divide_by_zero() -> None:
        divisor = int(measure(qubit()).read())
        result("quotient", 5 // divisor)

    with pytest.raises(RuntimeError, match="Attempted division by 0"):
        sim(Guppy(divide_by_zero)).classical(pecos.selene_engine()).qubits(1).run(1)


def test_mixed_types_fail() -> None:
    """Floats conflict with an established bool tag."""

    @guppy
    def mixed_results() -> None:
        result("mixed", measure(qubit()).read())
        result("mixed", 2.5)

    with pytest.raises(RuntimeError, match=r"mixed.*bool.*f64"):
        sim(Guppy(mixed_results)).classical(pecos.selene_engine()).qubits(1).run(1)


@pytest.mark.parametrize("signal", [0, 3, 1000])
def test_exit_preserves_results_and_continues_shots(signal: int) -> None:
    """Normal exits retain this shot's results and allow subsequent shots."""

    @guppy
    def exit_program() -> None:
        result("before", measure(qubit()).read())
        result("value", 7)
        guppy_exit("done", signal)

    data = sim(Guppy(exit_program)).classical(pecos.selene_engine()).qubits(1).run(4).to_dict()
    assert data == {"before": [0] * 4, "value": [7] * 4}


def test_exit_above_range_and_explicit_panic_fail() -> None:
    """The raw code, rather than the source function name, determines failure."""

    @guppy
    def high_exit() -> None:
        result("before", measure(qubit()).read())
        guppy_exit("high exit", 1001)

    @guppy
    def explicit_panic() -> None:
        result("before", measure(qubit()).read())
        panic("explicit panic", 1)

    for program, message in [(high_exit, "high exit"), (explicit_panic, "explicit panic")]:
        with pytest.raises(RuntimeError, match=rf"code=1001.*{message}"):
            sim(Guppy(program)).classical(pecos.selene_engine()).qubits(1).run(2)


def test_installed_runtime_plugin_forwards_numeric_outputs() -> None:
    """Exercise the installed Base QIS plugin through the C shim's plain ABI."""
    import ctypes
    import json
    import os

    from selene_base_qis_plugin import BaseQISInterface

    # Load the same process-wide FFI and shim libraries used by the QIS route.
    sim(Guppy(typed_results)).classical(pecos.selene_engine()).qubits(1).run(1)
    ffi = ctypes.CDLL(None)
    plugin = ctypes.CDLL(str(BaseQISInterface().library_file), mode=os.RTLD_LAZY)
    create = ffi.pecos_create_execution_context
    create.restype = ctypes.c_void_p
    register = ffi.pecos_register_execution_context
    register.argtypes = [ctypes.c_void_p]
    register.restype = None
    destroy = ffi.pecos_destroy_execution_context
    destroy.argtypes = [ctypes.c_void_p]
    destroy.restype = None
    get_json = ffi.pecos_get_named_results_json
    get_json.restype = ctypes.c_void_p
    free_json = ffi.pecos_free_named_results_json
    free_json.argtypes = [ctypes.c_void_p]
    free_json.restype = None
    context = create()
    register(context)
    try:
        for symbol, tag, value, ctype in [
            ("print_int", b"\x01i", 7, ctypes.c_int64),
            ("print_uint", b"\x01u", 11, ctypes.c_uint64),
            ("print_float", b"\x01f", 2.5, ctypes.c_double),
        ]:
            output = getattr(plugin, symbol)
            output.argtypes = [ctypes.c_char_p, ctypes.c_uint64, ctype]
            output.restype = None
            output(tag, 1, value)
        ptr = get_json()
        assert ptr
        try:
            assert json.loads(ctypes.string_at(ptr)) == {
                "i": {"type": "i64", "values": [7]},
                "u": {"type": "u64", "values": [11]},
                "f": {"type": "f64", "values": [4612811918334230528]},
            }
        finally:
            free_json(ptr)
    finally:
        register(None)
        destroy(context)


def test_exiting_shots_release_program_allocations() -> None:
    """Two hundred exits leave no growth in live program allocations."""
    import ctypes

    # Exercise the program allocator explicitly: fixed Guppy arrays can be
    # lowered without heap storage, which would make this leak test vacuous.
    allocating_exit = Qis(r"""
        @tag = private constant [7 x i8] c"\06before"
        @message = private constant [5 x i8] c"\04done"
        declare ptr @heap_alloc(i64)
        declare void @print_int(ptr, i64, i64)
        declare void @panic(i32, ptr)
        define void @main() {
            %allocation = call ptr @heap_alloc(i64 4096)
            store i64 3, ptr %allocation
            %value = load i64, ptr %allocation
            call void @print_int(ptr @tag, i64 6, i64 %value)
            call void @panic(i32 3, ptr @message)
            ret void
        }
    """)

    # Initialize the same process-wide FFI library the simulation uses.
    sim(Guppy(typed_results)).classical(pecos.selene_engine()).qubits(1).run(1)
    ffi = ctypes.CDLL(None)
    live = ffi.pecos_get_live_allocation_count
    live.restype = ctypes.c_size_t
    total = ffi.pecos_get_total_allocation_count
    total.restype = ctypes.c_size_t
    before = live()
    allocations_before = total()
    data = sim(allocating_exit).classical(pecos.selene_engine()).qubits(1).run(200).to_dict()
    after = live()
    assert data == {"before": [3] * 200}
    assert total() - allocations_before >= 200
    assert before == after == 0
    print(f"Live allocations before 200 exiting shots: {before}; after: {after}")


def test_branch_dependent_types_name_register_shot_and_types() -> None:
    """Cross-shot type mismatches identify the register, shot index, and types."""

    @guppy
    def mixed_branches() -> None:
        q = qubit()
        h(q)
        if measure(q).read():
            result("v", True)
        else:
            result("v", 7)

    shots = sim(Guppy(mixed_branches)).classical(pecos.selene_engine()).qubits(1).seed(42).run(32)
    with pytest.raises(RuntimeError, match=r"Register 'v' at shot 1: expected U32, received I64") as error:
        shots.to_dict()
    print(str(error.value))
