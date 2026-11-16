"""Physical-outcome witnesses for post-scheduling byte-shadow provenance."""

from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pecos
import pecos_rslib
import pytest

DECLARATIONS = """
@label = private constant [3 x i8] c"out"
declare i64 @___qalloc()
declare i64 @___lazy_measure(i64)
declare i64 @___lazy_measure_leaked(i64)
declare i64 @___read_future_uint(i64)
declare i1 @___read_future_bool(i64)
declare void @___qfree(i64)
declare void @print_bool_selene(ptr, i64, i1)
declare void @print_bool_arr_selene(ptr, i64, ptr, i64)
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1 immarg)
declare void @llvm.memset.p0.i64(ptr, i8, i64, i1 immarg)
declare ptr @heap_alloc(i64)
declare void @heap_free(ptr)
"""
READS = """
%q = call i64 @___qalloc()
%r = call i64 @___qalloc()
%f = call i64 @___lazy_measure(i64 %q)
%g = call i64 @___lazy_measure(i64 %r)
%a = call i1 @___read_future_bool(i64 %f)
%b = call i1 @___read_future_bool(i64 %g)
"""


def emit(value):
    return f"call void @print_bool_selene(ptr @label, i64 3, i1 {value})\n"


def program(body, helper="", *, main=False):
    entry = "define void @main()" if main else "define i64 @qmain(i64 %shot)"
    ret = "ret void" if main else "ret i64 0"
    return DECLARATIONS + helper + f"\n{entry} {{\nentry:\n{body}\n{ret}\n}}"


def capture(source, seed=92733):
    trace = (
        pecos.sim(pecos.Qis(source) if isinstance(source, str) else source)
        .classical(pecos.selene_engine("selene_simple_runtime"))
        .quantum(pecos_rslib.coin_toss())
        .qubits(8)
        .workers(1)
        .seed(seed)
        .capture_operation_trace(shots=64)
    )
    shots = defaultdict(lambda: {"records": [], "outcomes": None})
    for chunk in trace:
        s = shots[chunk["engine_trace_id"], chunk["shot_index"]]
        s["records"].extend(chunk["named_result_traces"])
        if chunk["stage"] == "trace_complete":
            s["outcomes"] = {int(k): bool(v) for k, v in chunk["measurement_results"].items()}
    assert len(shots) == 64
    for shot in shots.values():
        assert shot["outcomes"] is not None
        for record in shot["records"]:
            assert record["provenance"] is not None
            for value, p in zip(record["values"], record["provenance"], strict=True):
                if p["class"] == "measurement":
                    assert value == shot["outcomes"][p["id"]]
    return list(shots.values())


def identities(shot):
    return [p for r in shot["records"] for p in r["provenance"]]


@pytest.mark.parametrize("main", [False, True])
def test_reversed_separated_reads(main):
    for s in capture(program(READS + emit("%b") + emit("%a"), main=main)):
        assert identities(s) == [{"class": "measurement", "id": 1}, {"class": "measurement", "id": 0}]


def test_arithmetic_has_no_identity():
    body = READS + "%xor = xor i1 %a, %b\n%not = xor i1 %a, true\n" + emit("%xor") + emit("%not")
    for s in capture(program(body)):
        assert identities(s) == [{"class": "derived"}] * 2
        assert s["records"][0]["values"] == [s["outcomes"][0] ^ s["outcomes"][1]]
        assert s["records"][1]["values"] == [not s["outcomes"][0]]


def test_wide_copy_and_memcpy():
    body = READS + """
%x = alloca [2 x i8]
%y = alloca [2 x i8]
%z = alloca [2 x i8]
%p = getelementptr i8, ptr %x, i64 1
store i1 %a, ptr %x
store i1 %b, ptr %p
%wide = load i16, ptr %x, align 1
store i16 %wide, ptr %y, align 1
call void @llvm.memcpy.p0.p0.i64(ptr %z, ptr %y, i64 2, i1 false)
call void @print_bool_arr_selene(ptr @label, i64 3, ptr %z, i64 2)
"""
    for s in capture(program(body)):
        assert identities(s) == [{"class": "measurement", "id": 0}, {"class": "measurement", "id": 1}]


def test_fixed_integer_vector_copy():
    body = READS + """
%x = alloca [2 x i64]
%y = alloca [2 x i64]
%p = getelementptr i64, ptr %x, i64 1
%wide = zext i1 %a to i64
store i64 %wide, ptr %x
store i64 0, ptr %p
%vector = load <2 x i64>, ptr %x, align 8
store <2 x i64> %vector, ptr %y, align 8
%copied = load i64, ptr %y
%bit = trunc i64 %copied to i1
""" + emit("%bit")
    for shot in capture(program(body)):
        assert identities(shot) == [{"class": "measurement", "id": 0}]
        assert shot["records"][0]["values"] == [shot["outcomes"][0]]


def test_vector_arithmetic_rejects():
    body = READS + """
%x = alloca [2 x i64]
store <2 x i64> zeroinitializer, ptr %x
%v = load <2 x i64>, ptr %x
%sum = add <2 x i64> %v, %v
store <2 x i64> %sum, ptr %x
"""
    with pytest.raises(Exception, match="non-constant vector value"):
        capture(program(body))


def test_helper_two_calls_and_aggregate():
    helper = """define fastcc {i1, i1} @helper(i1 noundef %a, i1 %b) {
entry:
%x = insertvalue {i1, i1} poison, i1 %b, 0
%y = insertvalue {i1, i1} %x, i1 %a, 1
ret {i1, i1} %y
}
"""
    body = READS + """
%x = call fastcc {i1, i1} @helper(i1 %a, i1 %b)
%y = call fastcc {i1, i1} @helper(i1 %b, i1 %a)
%v = extractvalue {i1, i1} %x, 0
%w = extractvalue {i1, i1} %y, 0
""" + emit("%v") + emit("%w")
    source = program(body, helper).replace("@helper", '@"tail call helper"')
    for s in capture(source):
        assert identities(s) == [{"class": "measurement", "id": 1}, {"class": "measurement", "id": 0}]


def test_loop_measurements():
    body = (
        """br label %loop
loop:
%i = phi i64 [0, %entry], [%next, %loop]
%q = call i64 @___qalloc()
%f = call i64 @___lazy_measure(i64 %q)
%a = call i1 @___read_future_bool(i64 %f)
"""
        + emit("%a")
        + """call void @___qfree(i64 %q)
%next = add i64 %i, 1
%done = icmp eq i64 %next, 3
br i1 %done, label %end, label %loop
end:
"""
    )
    for s in capture(program(body)):
        assert identities(s) == [{"class": "measurement", "id": i} for i in range(3)]


def test_control_flag_is_sticky_after_branch():
    body = READS + emit("%a") + """br i1 %a, label %yes, label %no
yes:
br label %end
no:
br label %end
end:
""" + emit("%b")
    for s in capture(program(body)):
        assert [r["control_dependent"] for r in s["records"]] == [False, True]


def test_address_dependency():
    body = READS + """
%x = alloca [2 x i8]
%p = getelementptr i8, ptr %x, i64 1
store i1 false, ptr %x
store i1 true, ptr %p
%index = zext i1 %a to i64
%chosen = getelementptr i8, ptr %x, i64 %index
%v = load i1, ptr %chosen
""" + emit("%v")
    for s in capture(program(body)):
        assert identities(s) == [{"class": "derived"}]
        assert s["records"][0]["control_dependent"] is True
        assert s["records"][0]["values"] == [s["outcomes"][0]]


def test_unlisted_scalar_external_is_unknown(monkeypatch):
    from pecos import testing
    from pecos.qec.dem import _generator_certified_result_traces
    from pecos.qec.dem_spec import _index_result_traces
    from pecos.qec.surface.decode import _index_surface_result_trace_ids, _validate_surface_array_slot_provenance

    monkeypatch.setattr(testing, "measurement_ids_in_execution_order", lambda _circuit: [0, 1])
    monkeypatch.setattr("pecos.qec.dem._validated_source_measurement_ids", lambda _circuit: [0])
    body = READS + "%wide = zext i1 %a to i32\n%x = call i32 @abs(i32 %wide)\n%v = trunc i32 %x to i1\n" + emit("%v")
    for s in capture(program(body, "declare i32 @abs(i32)\n")):
        assert identities(s) == [{"class": "unknown"}]
        assert s["records"][0]["values"] == [s["outcomes"][0]]
        traces = s["records"]
        monkeypatch.setattr(
            testing,
            "_trace_program_to_tick_circuit_with_result_traces",
            lambda *_, traces=traces: (None, traces),
        )
        consumers = [
            lambda traces=traces: _index_result_traces(traces),
            lambda traces=traces: _index_surface_result_trace_ids(traces),
            lambda traces=traces: _validate_surface_array_slot_provenance(traces, {}),
            lambda: testing.measurement_partition_from_trace(None, 2, {}),
            lambda traces=traces: _generator_certified_result_traces(
                [("out", 0)],
                SimpleNamespace(result_tags=({}, 1)),
                None,
                traces,
                required_tags=[],
            ),
        ]
        for consume in consumers:
            with pytest.raises(ValueError, match=r"out.*occurrence 0.*unknown"):
                consume()


def test_unlisted_pointer_external_rejects():
    with pytest.raises(Exception, match=r"unlisted external.*mystery_ptr.*pointer"):
        capture(program("call void @mystery_ptr(ptr null)", "declare void @mystery_ptr(ptr)\n"))


def test_memset_copies_fill_shadow():
    source = (Path(__file__).parent / "fixtures/provenance/memset.ll").read_text()
    for s in capture(source):
        assert identities(s) == [{"class": "measurement", "id": 0}]
        assert s["records"][0]["values"] == [s["outcomes"][0]]


def test_leakage_read_does_not_poison_seed_stash():
    body = """%q = call i64 @___qalloc()
%f = call i64 @___lazy_measure_leaked(i64 %q)
%leaked = call i64 @___read_future_uint(i64 %f)
%g = call i64 @___lazy_measure(i64 %q)
%a = call i1 @___read_future_bool(i64 %g)
""" + emit("%a")
    for s in capture(program(body)):
        assert identities(s) == [{"class": "measurement", "id": 1}]


def test_qir_zero_one_comparisons():
    helper = """declare i32 @__quantum__qis__m__body(i64, i64)
declare i32 @__quantum__rt__result_get_one(i64)
"""
    body = """%m = call i32 @__quantum__qis__m__body(i64 0, i64 7)
%old32 = call i32 @__quantum__rt__result_get_one(i64 7)
%copy = icmp ne i32 %old32, 0
%not = icmp eq i32 %old32, 0
""" + emit("%copy") + emit("%not")
    for s in capture(program(body, helper)):
        assert identities(s) == [{"class": "measurement", "id": 0}, {"class": "derived"}]
        assert s["records"][1]["values"] == [not s["outcomes"][0]]


def test_guppy_separated_outputs():
    from guppylang import guppy
    from guppylang.std.builtins import result
    from guppylang.std.quantum import measure, qubit

    @guppy
    def circuit() -> None:
        first = measure(qubit()).read()
        second = measure(qubit()).read()
        result("second", second)
        result("first", first)

    shots = capture(pecos.Guppy(circuit))
    for s in shots:
        assert len(identities(s)) == 2
        assert {p["id"] for p in identities(s)} == {0, 1}
        assert all(p["class"] == "measurement" for p in identities(s))


def test_guppy_measure_array():
    from guppylang import guppy
    from guppylang.std.builtins import array, result
    from guppylang.std.quantum import collect_measurements, measure_array, qubit

    @guppy
    def circuit() -> None:
        qs = array(qubit() for _ in range(3))
        result("array", collect_measurements(measure_array(qs)))

    for s in capture(pecos.Guppy(circuit)):
        assert len(identities(s)) == 3
        assert {p["id"] for p in identities(s)} == {0, 1, 2}
        assert all(p["class"] == "measurement" for p in identities(s))


def test_recursive_calls_keep_shadow_frames():
    helper = """define i1 @recurse(i1 %a, i1 %b, i64 %depth) {
entry:
%done = icmp eq i64 %depth, 0
br i1 %done, label %base, label %step
base:
ret i1 %a
step:
%next = sub i64 %depth, 1
%r = call i1 @recurse(i1 %b, i1 %a, i64 %next)
ret i1 %r
}
"""
    body = READS + "%x = call i1 @recurse(i1 %a, i1 %b, i64 2)\n" + emit("%x")
    for s in capture(program(body, helper)):
        assert identities(s) == [{"class": "measurement", "id": 0}]


def test_selene_seed_nonzero_byte():
    helper = "declare i64 @selene_future_read_bool(ptr, i64)\n"
    body = """%q = call i64 @___qalloc()
%f = call i64 @___lazy_measure(i64 %q)
%packed = call i64 @selene_future_read_bool(ptr null, i64 %f)
%shifted = lshr i64 %packed, 32
%v = trunc i64 %shifted to i1
""" + emit("%v")
    for s in capture(program(body, helper)):
        assert identities(s) == [{"class": "measurement", "id": 0}]


def test_load_comparison_is_not_zero_one_proof():
    body = READS + """%x = alloca i32
store i32 0, ptr %x
store i1 %a, ptr %x
%p = getelementptr i8, ptr %x, i64 1
store i1 true, ptr %p
%wide = load i32, ptr %x
%v = icmp ne i32 %wide, 0
""" + emit("%v")
    for s in capture(program(body)):
        assert identities(s) == [{"class": "derived"}]
        assert s["records"][0]["values"] == [True]


def test_heap_and_reexecuted_alloca_are_fresh():
    # The helper's stack slot is reused across calls from the loop. Each alloca
    # execution must discard the previous invocation's shadow.
    helper = """define void @output_slot(i1 %m, i1 %first) {
entry:
%slot = alloca i8
br i1 %first, label %write, label %join
write:
store i1 %m, ptr %slot
br label %join
join:
%v = load i1, ptr %slot
""" + emit("%v") + "ret void\n}\n"
    body = """br label %loop
loop:
%i = phi i64 [0, %entry], [%next, %loop]
%q = call i64 @___qalloc()
%f = call i64 @___lazy_measure(i64 %q)
%m = call i1 @___read_future_bool(i64 %f)
%first = icmp eq i64 %i, 0
call void @output_slot(i1 %m, i1 %first)
call void @___qfree(i64 %q)
%next = add i64 %i, 1
%done = icmp eq i64 %next, 2
br i1 %done, label %end, label %loop
end:
"""
    for shot in capture(program(body, helper)):
        assert identities(shot) == [{"class": "measurement", "id": 0}, {"class": "unknown"}]


def test_heap_store_on_one_branch_resets_between_shots():
    body = READS + emit("%b") + """%p = call ptr @heap_alloc(i64 1)
br i1 %a, label %write, label %join
write:
store i1 %a, ptr %p
br label %join
join:
%v = load i1, ptr %p
""" + emit("%v")
    for s in capture(program(body)):
        expected = {"class": "measurement", "id": 0} if s["outcomes"][0] else {"class": "unknown"}
        assert identities(s) == [{"class": "measurement", "id": 1}, expected]
        assert [r["control_dependent"] for r in s["records"]] == [False, True]


def test_odd_width_store_preserves_adjacent_measurement_byte():
    body = (
        READS
        + """%x = alloca i32
%p = getelementptr i8, ptr %x, i64 3
store i1 %b, ptr %p
%a24 = zext i1 %a to i24
store i24 %a24, ptr %x, align 1
%v = load i1, ptr %p
"""
        + emit("%v")
        + """store <3 x i8> <i8 0, i8 0, i8 0>, ptr %x, align 1
%after_vector = load i1, ptr %p
"""
        + emit("%after_vector")
    )
    for s in capture(program(body)):
        assert identities(s) == [{"class": "measurement", "id": 1}] * 2


@pytest.mark.parametrize("operation", ["memset", "memcpy", "memmove"])
@pytest.mark.parametrize("base", [0, 1])
def test_measurement_dependent_memory_length(operation, base):
    body = READS + """
%x = alloca [2 x i8]
%source = alloca [2 x i8]
%second = getelementptr i8, ptr %x, i64 1
%source_second = getelementptr i8, ptr %source, i64 1
store i8 0, ptr %x
store i8 0, ptr %second
store i1 %b, ptr %source
store i8 1, ptr %source_second
%count = zext i1 %a to i64
"""
    if base:
        body += "%length = add i64 %count, 1\n"
    else:
        # Reproduce the zero-length case: untouched bytes retain their old class.
        body += "store i8 1, ptr %source\n"
    length = "%length" if base else "%count"
    if operation == "memset":
        body += f"call void @llvm.memset.p0.i64(ptr %x, i8 1, i64 {length}, i1 false)\n"
    else:
        body += f"call void @llvm.{operation}.p0.p0.i64(ptr %x, ptr %source, i64 {length}, i1 false)\n"
    body += f"%value = load i1, ptr {'%second' if base else '%x'}\n" + emit("%value")
    if base:
        body += "%first = load i1, ptr %x\n" + emit("%first")
    source = program(body)
    if operation == "memmove":
        source = "declare void @llvm.memmove.p0.p0.i64(ptr, ptr, i64, i1 immarg)\n" + source
    for shot in capture(source):
        measurement = shot["outcomes"][0]
        assert shot["records"][0]["values"] == [measurement]
        assert shot["records"][0]["provenance"] == [{"class": "derived" if measurement else "constant"}]
        assert all(record["control_dependent"] for record in shot["records"])
        if base:
            assert shot["records"][1]["values"] == [True if operation == "memset" else shot["outcomes"][1]]
            assert shot["records"][1]["provenance"] == [{"class": "derived"}]


@pytest.mark.parametrize("allocation", ["alloca", "heap_alloc"])
def test_measurement_dependent_allocation_count(allocation):
    body = READS + "%count = zext i1 %a to i64\n%length = add i64 %count, 1\n"
    body += "%x = alloca i8, i64 %length\n" if allocation == "alloca" else "%x = call ptr @heap_alloc(i64 %length)\n"
    body += "store i1 %b, ptr %x\n%value = load i1, ptr %x\n" + emit("%value")
    for shot in capture(program(body)):
        assert shot["records"][0]["values"] == [shot["outcomes"][1]]
        assert identities(shot) == [{"class": "derived"}]
        assert shot["records"][0]["control_dependent"]


@pytest.mark.parametrize("dense", [False, True])
def test_measurement_dependent_array_length(dense):
    body = READS + """
%x = alloca [2 x i8]
%second = getelementptr i8, ptr %x, i64 1
store i1 %a, ptr %x
store i1 %b, ptr %second
%count = zext i1 %a to i64
%length = add i64 %count, 1
"""
    if dense:
        body += """
%descriptor = alloca {i32, i32, ptr, ptr}
store {i32, i32, ptr, ptr} zeroinitializer, ptr %descriptor
%length32 = trunc i64 %length to i32
%data_field = getelementptr {i32, i32, ptr, ptr}, ptr %descriptor, i32 0, i32 2
store i32 %length32, ptr %descriptor
store ptr %x, ptr %data_field
call void @print_bool_arr(ptr @dense_label, i64 3, ptr %descriptor)
"""
    else:
        body += "call void @print_bool_arr_selene(ptr @label, i64 3, ptr %x, i64 %length)\n"
    source = program(body)
    if dense:
        source = (
            '@dense_label = private constant [4 x i8] c"\\03out"\ndeclare void @print_bool_arr(ptr, i64, ptr)\n'
            + source
        )
    for shot in capture(source):
        expected = [shot["outcomes"][0]] + ([shot["outcomes"][1]] if shot["outcomes"][0] else [])
        assert shot["records"][0]["values"] == expected
        assert identities(shot) == [{"class": "derived"}] * len(expected)
        assert shot["records"][0]["control_dependent"]


@pytest.mark.parametrize("kind", ["int", "uint", "float"])
@pytest.mark.parametrize("dense", [False, True])
def test_numeric_array_range_sets_flag_without_binding(kind, dense):
    body = READS + """
%x = alloca i64
store i64 0, ptr %x
%length = zext i1 %a to i64
"""
    if dense:
        body += """
%descriptor = alloca {i32, i32, ptr, ptr}
store {i32, i32, ptr, ptr} zeroinitializer, ptr %descriptor
%length32 = trunc i64 %length to i32
%data_field = getelementptr {i32, i32, ptr, ptr}, ptr %descriptor, i32 0, i32 2
store i32 %length32, ptr %descriptor
store ptr %x, ptr %data_field
"""
        body += f"call void @print_{kind}_arr(ptr @numbers, i64 4, ptr %descriptor)\n"
        declarations = f'declare void @print_{kind}_arr(ptr, i64, ptr)\n@numbers = constant [5 x i8] c"\\04nums"\n'
    else:
        body += f"call void @print_{kind}_arr_selene(ptr @numbers, i64 4, ptr %x, i64 %length)\n"
        declarations = (
            f'declare void @print_{kind}_arr_selene(ptr, i64, ptr, i64)\n@numbers = constant [4 x i8] c"nums"\n'
        )
    for shot in capture(declarations + program(body + emit("%b"))):
        assert shot["records"][0]["values"] == [shot["outcomes"][1]]
        assert identities(shot) == [{"class": "measurement", "id": 1}]
        assert shot["records"][0]["control_dependent"]
