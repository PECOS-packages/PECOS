"""Lowered-program ordering and narrow scalar provenance certification."""

from pathlib import Path

import pytest
from guppylang import guppy
from guppylang.std.builtins import result
from guppylang.std.quantum import measure, qubit
from pecos.compilation_pipeline import compile_guppy_to_hugr
from pecos.qec._qis_result_tags import _CertificationError, extract_result_tag_measurements
from pecos_rslib.hugr_lowering import compile_hugr_to_qis

FIXTURES = Path(__file__).parent / "fixtures" / "result_tags"
TAG = '@tag = private constant [12 x i8] c"\\0BUSER:BOOL:t"'
MEASURE = "%f = call i64 @___lazy_measure(i64 0)"
READ = "%v = call i1 @___read_future_bool(i64 %f)"
PRINT = "call void @print_bool(ptr @tag, i64 11, i1 %v)"


@guppy
def _backslash_tag() -> None:
    result("back\\slash", measure(qubit()).read())


def test_compiled_backslash_tag() -> None:
    qis = compile_hugr_to_qis(compile_guppy_to_hugr(_backslash_tag))
    assert extract_result_tag_measurements(qis) == ({"back\\slash": [0]}, 1)


def _program(body: str, extra: str = "") -> str:
    return (
        f"{TAG}\ndefine void @qmain(i1 %cond, ptr %fn) {{\n{body}\n}}\n{extra}\n"
        "declare void @panic(i32, ptr)\ndeclare i1 @opaque()\n"
    )


@pytest.mark.parametrize(
    ("name", "expected", "count"),
    [
        ("scrambled", {"tag_a": [0], "tag_b": [1], "tag_c": [2]}, 3),
        ("inlined", {"a": [1], "b": [0], "c": [2]}, 3),
        ("computed", {"const": [None], "eq": [None]}, 2),
        ("arr", {"pair": [None]}, 2),
        ("looped", {"synx": [0, 1, 2]}, 3),
        ("funcdecl", {}, 1),
    ],
)
def test_compiled_fixture(name: str, expected: dict[str, list[int | None]], count: int) -> None:
    tags, actual_count = extract_result_tag_measurements(compile_hugr_to_qis((FIXTURES / f"{name}.hugr").read_bytes()))
    assert tags == expected
    assert list(tags) == sorted(tags)
    assert actual_count == count


def test_block_layout_does_not_determine_ordinals() -> None:
    ir = _program(f"""
entry:
  br label %first
last:
  %g = call i64 @___lazy_measure(i64 1)
  {READ}
  {PRINT}
  ret void
first:
  {MEASURE}
  br label %last
""")
    assert extract_result_tag_measurements(ir) == ({"t": [0]}, 2)


def test_label_and_instruction_on_same_line() -> None:
    ir = _program(f"""
entry:
  {MEASURE}
  {READ}
  br label %p
p: {PRINT}
  %g = call i64 @___lazy_measure(i64 1)
  %w = call i1 @___read_future_bool(i64 %g)
  call void @print_bool(ptr @tag, i64 11, i1 %w)
  ret void
""")
    assert extract_result_tag_measurements(ir) == ({"t": [0, 1]}, 2)


@pytest.mark.parametrize(
    "terminator",
    [
        "ret void",
        "br label %entry",
        "switch i32 0, label %entry []",
        "unreachable",
        "%r = invoke i32 @foo() to label %entry unwind label %entry",
        "%r = callbr i32 @foo() to label %entry [label %entry]",
        "indirectbr ptr %fn, [label %entry]",
        "resume { ptr, i32 } undef",
        "%r = catchswitch within none [label %entry] unwind to caller",
        "cleanupret from %pad unwind to caller",
        "catchret from %pad to label %entry",
    ],
)
@pytest.mark.parametrize("dead", [False, True])
def test_terminator_must_be_last(terminator: str, *, dead: bool) -> None:
    opcode = terminator.removeprefix("%r = ").split()[0]
    prefix = "start:\nret void\n" if dead else ""
    ir = _program(f"{prefix}entry:\n{terminator}\n{MEASURE}\nret void")
    with pytest.raises(_CertificationError, match=f"terminator {opcode} is not last in block %entry"):
        extract_result_tag_measurements(ir)


@pytest.mark.parametrize("body", [f"entry:\n{MEASURE}", "entry:", "entry:\nnext:\nret void"])
@pytest.mark.parametrize("dead", [False, True])
def test_block_requires_terminator(body: str, *, dead: bool) -> None:
    prefix = "start:\nret void\n" if dead else ""
    with pytest.raises(_CertificationError, match="block %entry has no terminator"):
        extract_result_tag_measurements(_program(prefix + body))


@pytest.mark.parametrize(
    "middle",
    [
        """br i1 %cond, label %left, label %right
left:
  br label %join
right:
  br label %join""",
        """br label %loop
loop:
  br i1 %cond, label %loop, label %join""",
        """switch i32 0, label %join [
    i32 1, label %left
    i32 2, label %right
  ]
left:
  br label %join
right:
  br label %join""",
        """br i1 %cond, label %left, label %right
left:
  br i1 %cond, label %right, label %join
right:
  br i1 %cond, label %left, label %join""",
    ],
    ids=["diamond", "loop", "switch", "irreducible"],
)
def test_measurement_free_control_flow(middle: str) -> None:
    ir = _program(f"""
entry:
  {MEASURE}
  {READ}
  {PRINT}
  {middle}
join:
  %g = call i64 @___lazy_measure(i64 1)
  %w = call i1 @___read_future_bool(i64 %g)
  call void @print_bool(ptr @tag, i64 11, i1 %w)
  ret void
""")
    assert extract_result_tag_measurements(ir) == ({"t": [0, 1]}, 2)


@pytest.mark.parametrize("operation", [MEASURE, PRINT.replace("%v", "true")], ids=["measurement", "print"])
def test_conditional_order_relevant_block_rejected(operation: str) -> None:
    ir = _program(f"""
entry:
  br i1 %cond, label %arm, label %join
arm:
  {operation}
  br label %join
join:
  ret void
""")
    with pytest.raises(ValueError, match="order-relevant block %arm does not dominate every reachable ret"):
        extract_result_tag_measurements(ir)


@pytest.mark.parametrize("operation", [MEASURE, PRINT.replace("%v", "true")], ids=["measurement", "print"])
def test_loop_order_relevant_block_rejected(operation: str) -> None:
    ir = _program(f"""
entry:
  br label %loop
loop:
  {operation}
  br i1 %cond, label %loop, label %done
done:
  ret void
""")
    with pytest.raises(ValueError, match="order-relevant block %loop is on a CFG cycle"):
        extract_result_tag_measurements(ir)


def test_measurement_must_dominate_all_returns() -> None:
    ir = _program(f"""
entry:
  br i1 %cond, label %measured, label %other
measured:
  {MEASURE}
  ret void
other:
  ret void
""")
    with pytest.raises(ValueError, match="%measured does not dominate every reachable ret"):
        extract_result_tag_measurements(ir)


def test_join_intersects_all_predecessor_dominators() -> None:
    # The parser visits right before left: choosing only the first predecessor
    # would falsely make right dominate join. Preserve this successor order.
    ir = _program(f"""
entry:
  br i1 %cond, label %left, label %right
left:
  br label %join
right:
  {MEASURE}
  br label %join
join:
  ret void
""")
    with pytest.raises(_CertificationError, match="%right does not dominate every reachable ret"):
        extract_result_tag_measurements(ir)


def test_measurement_dominates_several_returns() -> None:
    ir = _program(f"""
entry:
  {MEASURE}
  {READ}
  {PRINT}
  br i1 %cond, label %left, label %right
left:
  ret void
right:
  br i1 %cond, label %second, label %third
second:
  ret void
third:
  ret void
""")
    assert extract_result_tag_measurements(ir) == ({"t": [0]}, 1)


@pytest.mark.parametrize(
    ("name", "args"),
    [("__quantum__qis__m__body", "i64 0, i64 0"), ("__quantum__qis__mz__body", "i64 0")],
)
@pytest.mark.parametrize("conditional", [False, True])
def test_non_lazy_runtime_measurements_rejected(name: str, args: str, *, conditional: bool) -> None:
    call = f"%r = call i32 @{name}({args})"
    if conditional:
        body = f"entry:\nbr i1 %cond, label %arm, label %done\narm:\n{call}\nbr label %done\ndone:\nret void"
        reason = "order-relevant block %arm does not dominate every reachable ret"
    else:
        body = f"{call}\nret void"
        reason = f"unsupported measurement call @{name}"
    parameters = ", ".join(arg.split()[0] for arg in args.split(", "))
    with pytest.raises(_CertificationError, match=reason):
        extract_result_tag_measurements(_program(body, f"declare i32 @{name}({parameters})"))


@pytest.mark.parametrize("targets", ["label %panic, label %ok", "label %ok, label %panic"])
def test_panic_branches(targets: str) -> None:
    ir = _program(f"""
entry:
  br i1 %cond, {targets}
panic:
  call void @panic(i32 1001, ptr null)
  unreachable
ok:
  {MEASURE}
  {READ}
  {PRINT}
  ret void
""")
    assert extract_result_tag_measurements(ir) == ({"t": [0]}, 1)


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ("invoke void @foo() to label %ok unwind label %bad", "unsupported terminator invoke"),
        ("callbr void @foo() to label %ok [label %bad]", "unsupported terminator callbr"),
        ("indirectbr ptr %fn, [label %ok]", "unsupported terminator indirectbr"),
        ("call void %fn()\nret void", "indirect call"),
        ("call void @qmain(i1 true, ptr null)\nret void", "recursive call to @qmain"),
        (
            "%r = call i64 @__quantum__rt__result_allocate()\nret void",
            "result-allocating call @__quantum__rt__result_allocate",
        ),
        (MEASURE + "\nunreachable", "function has no reachable ret"),
        (
            PRINT.replace("%v", "true").replace("@tag", "%fn") + "\nret void",
            "result tag operand is not a module global",
        ),
        (PRINT.replace("%v", "true").replace("@tag", "@missing") + "\nret void", "missing tag global @missing"),
    ],
)
def test_unsupported_programs(body: str, reason: str) -> None:
    with pytest.raises(ValueError, match=f"^Cannot certify measurement order from the lowered program: {reason}$"):
        extract_result_tag_measurements(_program(body))


def test_defined_callee_counts_at_call_site_and_keeps_ssa_local() -> None:
    helper = f"define void @helper() {{\n{MEASURE}\n{READ}\n{PRINT}\nret void\n}}"
    ir = _program(
        f"""
  {MEASURE}
  call void @helper()
  {READ}
  {PRINT}
  call void @helper()
  ret void
""",
        helper,
    )
    assert extract_result_tag_measurements(ir) == ({"t": [1, 0, 2]}, 3)


def test_conditional_transitive_callee_rejected() -> None:
    helper = f"define void @helper() {{\n{MEASURE}\nret void\n}}"
    ir = _program(
        """
entry:
  br i1 %cond, label %arm, label %done
arm:
  call void @helper()
  br label %done
done:
  ret void
""",
        helper,
    )
    with pytest.raises(ValueError, match="%arm does not dominate every reachable ret"):
        extract_result_tag_measurements(ir)


@pytest.mark.parametrize(
    ("definition", "operand"),
    [
        ("", "true"),
        ("%selected = select i1 %cond, i1 %v, i1 false", "%selected"),
        ("br label %join\njoin:\n%merged = phi i1 [ %v, %entry ]", "%merged"),
        ("%loaded = load i1, ptr %fn", "%loaded"),
        ("%negated = xor i1 %v, true", "%negated"),
        ("%returned = call i1 @opaque()", "%returned"),
        ("", "%cond"),
    ],
    ids=["literal", "select", "phi", "load", "arithmetic", "call", "parameter"],
)
def test_only_exact_read_result_carries_provenance(definition: str, operand: str) -> None:
    ir = _program(f"entry:\n{MEASURE}\n{READ}\n{definition}\n{PRINT.replace('%v', operand)}\nret void")
    assert extract_result_tag_measurements(ir) == ({"t": [None]}, 1)


def test_leaked_measurement_counts_without_scalar_provenance() -> None:
    ir = _program(f"{MEASURE.replace('___lazy_measure', '___lazy_measure_leaked')}\n{READ}\n{PRINT}\nret void")
    assert extract_result_tag_measurements(ir) == ({"t": [None]}, 1)


def test_read_in_separate_block() -> None:
    ir = _program(f"""
entry:
  {MEASURE}
  br label %read
read:
  {READ}
  br label %print
print:
  {PRINT}
  ret void
""")
    assert extract_result_tag_measurements(ir) == ({"t": [0]}, 1)


@pytest.mark.parametrize("kind", ["bool", "int", "uint", "float"])
@pytest.mark.parametrize("suffix", ["", "_arr", "_selene", "_arr_selene"])
def test_all_runtime_print_exports(kind: str, suffix: str) -> None:
    prefix = kind.upper()
    plain = "_selene" in suffix
    array = "_arr" in suffix
    label = f"USER:{prefix}{'ARR' if array else ''}:t"
    data = label if plain else f"\\{len(label):02X}{label}"
    value = "ptr null" if array else {"bool": "i1 true", "int": "i64 1", "uint": "i64 1", "float": "double 1.0"}[kind]
    if plain and array:
        value += ", i64 0"
    ir = _program(f"call void @print_{kind}{suffix}(ptr @tag, i64 {len(label)}, {value})\nret void")
    ir = ir.replace(TAG, f'@tag = private constant [{len(label) + (not plain)} x i8] c"{data}"')
    assert extract_result_tag_measurements(ir) == ({"t": [None]}, 0)


def test_llvm_escaped_utf8_tag_and_runtime_length() -> None:
    # The runtime skips the prefix byte and uses the explicit length; neither
    # NULs nor colons terminate the tag, and the mangled global name is ignored.
    ir = _program(PRINT.replace("%v", "true").replace("i64 11", "i64 17") + "\nret void")
    ir = ir.replace(TAG, r'@tag = private constant [18 x i8] c"\00USER:BOOL:a\22\5C\C3\A9\00:"')
    assert extract_result_tag_measurements(ir) == ({'a"\\é\x00:': [None]}, 0)


@pytest.mark.parametrize(
    ("literal", "reason"),
    [
        (r"\0Bshort", "tag global is shorter"),
        (r"\0BNOPE:BOOL:t!", "tag label does not match"),
        (r"\0BUSER:BOOL:\FF", "invalid UTF-8 tag label"),
    ],
)
def test_invalid_tag_bytes(literal: str, reason: str) -> None:
    ir = _program(PRINT.replace("%v", "true") + "\nret void").replace(
        TAG,
        f'@tag = private constant [12 x i8] c"{literal}"',
    )
    with pytest.raises(ValueError, match=reason):
        extract_result_tag_measurements(ir)


def test_dead_blocks_and_uncalled_functions_are_ignored() -> None:
    ir = _program(
        f"entry:\nret void\ndead:\n{MEASURE}\ncall void %fn()\nret void",
        f"define void @unused() {{\n{MEASURE}\nret void\n}}",
    )
    assert extract_result_tag_measurements(ir) == ({}, 0)


def test_direct_call_with_explicit_function_type_and_pointer_attributes() -> None:
    ir = _program(
        f"{MEASURE}\n{READ}\n"
        "call void (ptr, i64, i1) @print_bool(ptr dereferenceable(12) @tag, i64 11, i1 %v)\nret void",
    )
    assert extract_result_tag_measurements(ir) == ({"t": [0]}, 1)


def test_alias_cannot_hide_measurements() -> None:
    ir = _program(
        "call void @alias()\nret void",
        f"@alias = alias void (), ptr @helper\ndefine void @helper() {{\n{MEASURE}\nret void\n}}",
    )
    with pytest.raises(ValueError, match="callee @alias is neither a function definition nor a declaration"):
        extract_result_tag_measurements(ir)


def test_comment_cannot_supply_tag_global_contents() -> None:
    ir = _program(PRINT.replace("%v", "true") + "\nret void")
    ir = ir.replace(TAG, '@tag = private global i8 0 ; constant [12 x i8] c"\\0BUSER:BOOL:t"')
    with pytest.raises(ValueError, match="missing tag global @tag"):
        extract_result_tag_measurements(ir)


def test_defined_always_panicking_helper_without_relevant_calls() -> None:
    ir = _program(
        f"""
entry:
  br i1 %cond, label %panic, label %ok
panic:
  call void @abort_helper()
  unreachable
ok:
  {MEASURE}
  {READ}
  {PRINT}
  ret void
""",
        """
define void @abort_helper() {
  call void @panic(i32 1001, ptr null)
  unreachable
}
""",
    )
    assert extract_result_tag_measurements(ir) == ({"t": [0]}, 1)


@pytest.mark.parametrize("cyclic_measurement", [False, True])
def test_deep_cfg_uses_iterative_traversals(*, cyclic_measurement: bool) -> None:
    # Exceed Python's recursion limit, with a multi-block SCC and scrambled layout.
    size = 2000
    blocks = []
    for index in range(size):
        operation = "%g = call i64 @___lazy_measure(i64 1)" if cyclic_measurement and index == 1000 else ""
        branch = f"br label %b{index + 1}" if index + 1 < size else "br i1 %cond, label %b0, label %done"
        blocks.append(f"b{index}:\n{operation}\n{branch}")
    ir = _program(
        f"entry:\n{MEASURE}\n{READ}\nbr label %b0\n" + "\n".join(reversed(blocks)) + f"\ndone:\n{PRINT}\nret void",
    )
    if cyclic_measurement:
        with pytest.raises(ValueError, match="order-relevant block %b1000 is on a CFG cycle"):
            extract_result_tag_measurements(ir)
    else:
        assert extract_result_tag_measurements(ir) == ({"t": [0]}, 1)


@pytest.mark.parametrize(
    "name",
    [
        "___lazy_measure",
        "___lazy_measure_leaked",
        "___read_future_bool",
        "__quantum__rt__result_allocate",
        "__quantum__qis__m__body",
        "__quantum__qis__mz__body",
        *(
            f"print_{kind}{suffix}"
            for kind in ("bool", "int", "uint", "float")
            for suffix in ("", "_arr", "_selene", "_arr_selene")
        ),
    ],
)
def test_module_cannot_define_runtime_calls(name: str) -> None:
    ir = _program("ret void", f"define void @{name}() {{\nret void\n}}")
    with pytest.raises(_CertificationError, match=f"module defines runtime call @{name}$"):
        extract_result_tag_measurements(ir)


@pytest.mark.parametrize("alias", [False, True])
def test_defined_print_cannot_redirect_tag(*, alias: bool) -> None:
    name = "helper" if alias else "print_bool"
    alias_definition = "@print_bool = alias void (ptr, i64, i1), ptr @helper" if alias else ""
    ir = _program(
        f"{MEASURE}\n{READ}\n{PRINT}\nret void",
        f"""@u = private constant [11 x i8] c"USER:BOOL:u"
{alias_definition}
define void @{name}(ptr %tag, i64 %len, i1 %value) {{
  call void @print_bool_selene(ptr @u, i64 11, i1 %value)
  ret void
}}""",
    )
    with pytest.raises(_CertificationError, match=r"module defines runtime call @print_bool$"):
        extract_result_tag_measurements(ir)


@pytest.mark.parametrize(
    ("ir", "reason"),
    [
        (_program("%f =\ncall i64 @___lazy_measure(i64 0)\nret void"), "incomplete instruction"),
        ("define void @qmain(", "Unterminated function parameter list"),
        (_program("tail\nret void"), "incomplete tail call"),
        (_program("%f = call i64 @___lazy_measure(i64)\nret void"), "missing operand"),
        (
            _program(
                "call void @f0()\nret void",
                "\n".join(f"define void @f{n}() {{\ncall void @f{n + 1}()\nret void\n}}" for n in range(2000))
                + "\ndefine void @f2000() {\nret void\n}",
            ),
            "call graph exceeds Python's recursion limit",
        ),
        (
            _program(f"{MEASURE}\n{READ}\ncall void @print_bool(ptr @tag, i64 {'1' * 5000}, i1 %v)\nret void"),
            "result tag length exceeds the supported integer range",
        ),
    ],
    ids=[
        "split-assignment",
        "unclosed-parameters",
        "incomplete-tail",
        "missing-operand",
        "deep-call-chain",
        "huge-tag-length",
    ],
)
def test_malformed_input_is_a_certification_error(ir: str, reason: str) -> None:
    with pytest.raises(_CertificationError, match=reason):
        extract_result_tag_measurements(ir)


@pytest.mark.parametrize("escape", [r"\n", r"\q", r"\0Z"])
def test_non_llvm_string_escapes_rejected(escape: str) -> None:
    ir = _program("ret void").replace(TAG, f'@tag = private constant [1 x i8] c"{escape}"')
    with pytest.raises(_CertificationError, match="invalid LLVM string escape"):
        extract_result_tag_measurements(ir)
