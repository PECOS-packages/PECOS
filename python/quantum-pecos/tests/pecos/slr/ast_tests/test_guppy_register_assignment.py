# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Whole-register assignments preserve Guppy's bool arrays and QASM bit order."""

import re
from dataclasses import replace

import numpy as np
import pytest
from pecos import Hugr, Qasm, selene_engine, sim, state_vector
from pecos.slr import Block, CReg, For, If, Main, Parallel, QAlloc, QReg, Repeat, Return, SlrConverter
from pecos.slr.ast import ast_to_guppy, slr_to_ast
from pecos.slr.ast.codegen.guppy import GuppyCodegenError
from pecos.slr.ast.nodes import (
    ArrayTypeExpr,
    AssignOp,
    BinaryExpr,
    BinaryOp,
    BitTypeExpr,
    BlockDecl,
    BlockInput,
    ForStmt,
    LiteralExpr,
    QubitTypeExpr,
    ResourceEffect,
    VarExpr,
)
from pecos.slr.qeclib import qubit as qb
from pecos.slr.vars import LoopVar


def _assert_matches_qasm(registers, operations, expected) -> None:
    """Observe every classical bit through a deterministic measured qubit.

    Selene's default records contain measurements, not returned bool arrays.
    Reading the registers into qubits makes the same SLR program observable
    in both engines, including post-assignment bit writes and source reuse.
    """
    size = sum(reg.size for reg in registers)
    q, observed = QReg("probe", size), CReg("observed", size)
    reads = []
    offset = 0
    for reg in registers:
        reads.extend(If(reg[index] == 1).Then(qb.X(q[offset + index])) for index in range(reg.size))
        offset += reg.size
    prog = Main(q, observed, *registers, *operations, *reads, qb.MeasureZ(q) > observed, Return(*registers, observed))
    converter = SlrConverter(prog)
    package = converter.hugr()
    selene = sim(Hugr(package.to_str().encode("utf-8"))).classical(selene_engine()).qubits(size).seed(42).run(2)
    records = selene.to_dict()
    qasm = sim(Qasm.from_string(converter.qasm())).quantum(state_vector()).seed(42).run(2).to_dict()
    for name, value in expected.items():
        assert qasm[name] == [value] * 2
    assert records == {
        f"measurement_{index}": [(value >> index) & 1 for value in qasm["observed"]] for index in range(size)
    }


@pytest.mark.parametrize(("width", "value"), [(1, 0), (1, 3), (4, 0), (4, 1), (4, 11), (4, 29), (4, -3)])
def test_literal_assignment_matches_qasm(width: int, value: int) -> None:
    c = CReg("c", width)
    # Start nonzero so assigning zero must actually clear the old value.
    operations = [*(c[index].set(1) for index in range(width)), c.set(value)]
    _assert_matches_qasm([c], operations, {"c": value & ((1 << width) - 1)})


@pytest.mark.parametrize("value", [0, 11, 15])
def test_register_copy_preserves_source_and_independent_bit_writes(value: int) -> None:
    c, d = CReg("c", 4), CReg("d", 4)
    operations = [
        *(d[index].set((value >> index) & 1) for index in range(4)),
        c.set(d),
        c.set(c),
        d[0].set(1),
        c[1].set(0),
    ]
    _assert_matches_qasm([c, d], operations, {"c": value & ~2, "d": value | 1})


@pytest.mark.parametrize("scope", ["block", "if", "repeat", "for", "parallel"])
def test_nested_literal_assignment_matches_qasm(scope: str) -> None:
    c = CReg("c", 4)
    assignment = c.set(11)
    nested = {
        "block": Block(assignment),
        "if": If(c[0] == 0).Then(assignment),
        "repeat": Repeat(2).block(assignment),
        "for": For("i", 0, 2).Do(assignment),
        "parallel": Parallel(assignment),
    }
    _assert_matches_qasm([c], [nested[scope]], {"c": 11})


@pytest.mark.parametrize(
    "form",
    ["register-add", "bit-add", "bit", "loop-add", "loop", "negate", "shift", "float"],
)
def test_unsupported_expression_fails_during_guppy_generation(form: str) -> None:
    c, d, i = CReg("c", 4), CReg("d", 4), LoopVar("i")
    expressions = {
        "register-add": d + 1,
        "bit-add": d[0] + 1,
        "bit": d[0],
        "loop-add": c + i,
        "loop": i,
        "negate": -d,
        "shift": d << 1,
        "float": 1.5,
    }
    operation = c.set(expressions[form])
    if form.startswith("loop"):
        operation = For(i, 0, 2).Do(operation)
    with pytest.raises(GuppyCodegenError, match=r"assignment to 'c'.*"):
        SlrConverter(Main(c, d, operation)).guppy()


@pytest.mark.parametrize("source_width", [2, 6])
def test_register_copy_rejects_different_widths(source_width: int) -> None:
    c, d = CReg("c", 4), CReg("d", source_width)
    with pytest.raises(GuppyCodegenError, match=r"'d'.*'c'.*equal widths"):
        SlrConverter(Main(c, d, c.set(d))).guppy()


@pytest.mark.parametrize("form", ["register-true", "register-false", "bit-true"])
def test_bool_literal_assignment_sets_the_low_bit(form: str) -> None:
    c = CReg("c", 4)
    operations, expected = {
        "register-true": ([c.set(True)], 1),
        "register-false": ([c.set(15), c.set(False)], 0),
        "bit-true": ([c[0].set(True)], 1),
    }[form]
    converter = SlrConverter(Main(c, *operations))
    if form == "register-true":
        assert "c = array(True, False, False, False)" in converter.guppy()
    _assert_matches_qasm([c], operations, {"c": expected})
    assert "True" not in converter.qasm()
    assert "False" not in converter.qasm()


@pytest.mark.parametrize("value", [True, False])
def test_numpy_bool_matches_python_bool(*, value: bool) -> None:
    c = CReg("c", 1)
    python = SlrConverter(Main(c, c[0].set(value)))
    numpy = SlrConverter(Main(c, c[0].set(np.bool_(value))))
    literal = slr_to_ast(Main(c, c[0].set(np.bool_(value)))).body[0].value
    assert isinstance(literal, LiteralExpr)
    assert literal.value is value
    assert numpy.qasm() == python.qasm()
    assert numpy.guppy() == python.guppy()


@pytest.mark.parametrize("operation", ["declare", "literal", "bool", "copy"])
@pytest.mark.parametrize("returned", [False, True])
def test_zero_width_classical_register_compiles(operation: str, *, returned: bool) -> None:
    z, source = CReg("z", 0), CReg("source", 0)
    operations = {"declare": [], "literal": [z.set(3)], "bool": [z.set(True)], "copy": [z.set(source)]}[operation]
    program = Main(QReg("q", 1), z, source, *operations, *([Return(z)] if returned else []))
    assert SlrConverter(program).hugr()


@pytest.mark.parametrize(
    ("form", "rendered"),
    [
        ("bit", "c[0] = d[1]"),
        ("and", "c[0] = (d[1] and d[2])"),
        ("eq", "c[0] = (d[1] == d[2])"),
        (
            "register-eq",
            "c[0] = (((1 if d[0] else 0) + (2 if d[1] else 0) + (4 if d[2] else 0) + (8 if d[3] else 0)) == 3)",
        ),
    ],
)
def test_bit_assignment_accepts_bit_values(form: str, rendered: str) -> None:
    c, d = CReg("c", 4), CReg("d", 4)
    values = {"bit": d[1], "and": d[1] & d[2], "eq": d[1] == d[2], "register-eq": d == 3}
    assert rendered in SlrConverter(Main(c, d, c[0].set(values[form]))).guppy()


@pytest.mark.parametrize(
    ("form", "message"),
    [
        ("register", "Unsupported single-bit assignment from 'd'. Index a bit of the register, such as d[0]."),
        ("bit-add", "Guppy operator ADD does not accept bool and bool operands"),
        ("float", "Unsupported single-bit assignment from '1.5'. Use 0, 1, a bit or a boolean expression."),
        ("register-and", "Guppy operator AND does not accept register and register operands"),
        ("negate", "Guppy operator NEG does not accept a bool operand"),
    ],
)
def test_bit_assignment_rejects_values_that_are_not_bits(form: str, message: str) -> None:
    c, d = CReg("c", 4), CReg("d", 4)
    values = {
        "register": d,
        "bit-add": d[0] + 1,
        "float": 1.5,
        "register-and": d & c,
        "negate": -d[0],
    }
    with pytest.raises(GuppyCodegenError, match=re.escape(message)):
        SlrConverter(Main(c, d, c[0].set(values[form]))).guppy()


def test_bit_assignment_rejects_a_loop_variable() -> None:
    c, i = CReg("c", 4), LoopVar("i")
    with pytest.raises(
        GuppyCodegenError,
        match=re.escape("Unsupported single-bit assignment from 'i'. Compare the loop variable, such as i == 1."),
    ):
        SlrConverter(Main(c, For(i, 0, 2).Do(c[i].set(i)))).guppy()


@pytest.mark.parametrize("method", ["guppy", "hugr"])
def test_loop_variable_shadowing_a_register_is_not_a_register_copy(method: str) -> None:
    c, d = CReg("c", 4), CReg("d", 4)
    converter = SlrConverter(Main(c, d, For("d", 0, 2).Do(c.set(1))))
    with pytest.raises(GuppyCodegenError, match=r"'d'.*classical register.*Rename the loop label"):
        getattr(converter, method)()


def test_loop_variable_is_not_a_register_copy() -> None:
    c, i = CReg("c", 4), LoopVar("i")
    with pytest.raises(GuppyCodegenError, match="assignment to 'c': 'i' is not a whole classical register"):
        SlrConverter(Main(c, For(i, 0, 2).Do(c.set(i)))).guppy()


@pytest.mark.parametrize("kind", ["qreg", "qalloc", "slot"])
def test_loop_label_rejects_quantum_bindings(kind: str) -> None:
    q = QAlloc(1, name="q") if kind == "qalloc" else QReg("q", 1)
    label = "q_0" if kind == "slot" else "q"
    with pytest.raises(GuppyCodegenError, match=f"'{label}'.*(quantum register/allocator|qubit binding).*Rename"):
        SlrConverter(Main(q, For(label, 0, 2).Do())).guppy()


@pytest.mark.parametrize("input_type", [BitTypeExpr(), QubitTypeExpr(), ArrayTypeExpr(element=QubitTypeExpr(), size=1)])
def test_loop_label_rejects_block_inputs(input_type) -> None:
    program = slr_to_ast(Main())
    block = BlockDecl(
        name="uses_input",
        inputs=(BlockInput(name="value", type_expr=input_type, effect=ResourceEffect.LIVE_PRESERVED),),
        body=(ForStmt(variable="value", start=LiteralExpr(value=0), stop=LiteralExpr(value=2), body=()),),
    )
    with pytest.raises(GuppyCodegenError, match=r"'value'.*block inputs.*Rename the loop label"):
        ast_to_guppy(replace(program, block_decls=(block,)))


@pytest.mark.parametrize("backend", ["qasm", "stim", "quantum_circuit"])
def test_string_loop_label_remains_an_unroll_label(backend: str) -> None:
    q = QReg("q", 1)
    program = Main(q, For("q", 0, 2).Do(qb.X(q[0])))
    getattr(SlrConverter(program), backend)()


def test_bit_assignment_integer_error_gives_fix() -> None:
    c = CReg("c", 1)
    with pytest.raises(GuppyCodegenError, match="Cannot render integer literal 2 as a Guppy bool; use 0 or 1"):
        SlrConverter(Main(c, c[0].set(2))).guppy()


def test_direct_ast_numbered_bit_target_is_a_bool() -> None:
    program = slr_to_ast(Main(CReg("c", 4)))
    program = replace(program, body=(AssignOp(target="c[2]", value=LiteralExpr(value=1)),))
    assert "c[2] = True" in ast_to_guppy(program)


@pytest.mark.parametrize(
    ("target", "value", "reason"),
    [
        ("missing", LiteralExpr(value=1), "Unresolved classical reference 'missing'"),
        ("c", VarExpr(name="missing"), "Unresolved classical reference 'missing'"),
        ("c", VarExpr(name="c[0]"), r"assignment to 'c': 'c\[0\]' is not a whole classical register"),
        ("c", BinaryExpr(op=BinaryOp.ADD, left=LiteralExpr(value=2), right=LiteralExpr(value=3)), "assignment to 'c'"),
    ],
)
def test_direct_ast_assignment_rejects_unsupported_forms(target, value, reason) -> None:
    program = slr_to_ast(Main(CReg("c", 4)))
    program = replace(program, body=(AssignOp(target=target, value=value),))
    with pytest.raises(GuppyCodegenError, match=reason):
        ast_to_guppy(program)


def test_loop_indexed_bit_literal_is_a_bool() -> None:
    # A symbolic bit target is a single-bit write; its literal must render as a Guppy bool.
    q, c, i = QReg("q", 1), CReg("c", 2), LoopVar("i")
    program = Main(q, c, For(i, 0, 2).Do(c[i].set(1)), Return(c))
    source = SlrConverter(program).guppy()
    assert "c[i] = True" in source
    assert SlrConverter(program).hugr()


@pytest.mark.parametrize("repetitions", [1, 2])
def test_repeat_counter_preserves_user_underscore_loop(repetitions: int) -> None:
    q, c, u = QReg("q", 1), CReg("c", 1), LoopVar("_")
    program = Main(
        q,
        c,
        For(u, 0, 3).Do(
            Repeat(repetitions).block(),
            c[0].set(u == 0),
            If(c[0] == 1).Then(qb.X(q[0])),
            qb.MeasureZ(q),
            qb.PZ(q),
        ),
    )
    # QASM cannot bind a referenced LoopVar. Observe each iteration directly in
    # Selene: only iteration zero should flip the freshly prepared qubit.
    package = SlrConverter(program).hugr()
    result = sim(Hugr(package.to_str().encode("utf-8"))).classical(selene_engine()).qubits(1).seed(42).run(2)
    assert result.to_dict() == {"measurement_0": [1, 1], "measurement_1": [0, 0], "measurement_2": [0, 0]}
