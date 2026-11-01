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

from dataclasses import replace

import pytest
from pecos import Hugr, Qasm, selene_engine, sim, state_vector
from pecos.slr import Block, CReg, For, If, Main, Parallel, QReg, Repeat, Return, SlrConverter
from pecos.slr.ast import ast_to_guppy, slr_to_ast
from pecos.slr.ast.codegen.guppy import GuppyCodegenError
from pecos.slr.ast.nodes import AssignOp, BinaryExpr, BinaryOp, LiteralExpr, VarExpr
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
    prog = Main(q, observed, *registers, *operations, *reads, qb.Measure(q) > observed, Return(*registers, observed))
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


def test_bool_literal_assignment_sets_the_low_bit() -> None:
    c = CReg("c", 4)
    assert "c = array(True, False, False, False)" in SlrConverter(Main(c, c.set(True))).guppy()


@pytest.mark.parametrize(
    ("form", "rendered"),
    [("bit", "c[0] = d[1]"), ("and", "c[0] = (d[1] and d[2])"), ("eq", "c[0] = (d[1] == d[2])")],
)
def test_bit_assignment_accepts_bit_values(form: str, rendered: str) -> None:
    c, d = CReg("c", 4), CReg("d", 4)
    values = {"bit": d[1], "and": d[1] & d[2], "eq": d[1] == d[2]}
    assert rendered in SlrConverter(Main(c, d, c[0].set(values[form]))).guppy()


@pytest.mark.parametrize("form", ["register", "bit-add", "float", "register-and", "register-eq", "negate"])
def test_bit_assignment_rejects_values_that_are_not_bits(form: str) -> None:
    c, d = CReg("c", 4), CReg("d", 4)
    values = {
        "register": d,
        "bit-add": d[0] + 1,
        "float": 1.5,
        "register-and": d & c,
        "register-eq": d == 3,
        "negate": -d[0],
    }
    with pytest.raises(GuppyCodegenError, match="Unsupported single-bit assignment"):
        SlrConverter(Main(c, d, c[0].set(values[form]))).guppy()


def test_bit_assignment_rejects_a_loop_variable() -> None:
    c, i = CReg("c", 4), LoopVar("i")
    with pytest.raises(GuppyCodegenError, match="Unsupported single-bit assignment"):
        SlrConverter(Main(c, For(i, 0, 2).Do(c[i].set(i)))).guppy()


def test_loop_variable_shadowing_a_register_is_not_a_register_copy() -> None:
    c, d = CReg("c", 4), CReg("d", 4)
    assert "c = array(True, False, False, False)" in SlrConverter(Main(c, d, For("d", 0, 2).Do(c.set(1)))).guppy()
    with pytest.raises(GuppyCodegenError, match="assignment to 'c': 'd' is not a whole classical register"):
        SlrConverter(Main(c, d, For("d", 0, 2).Do(c.set(d)))).guppy()


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
