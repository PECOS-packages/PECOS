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

"""Guppy code generation either rejects a classical expression or emits one Guppy accepts."""

from dataclasses import replace
from itertools import product

import numpy as np
import pytest
from pecos import Hugr, selene_engine, sim
from pecos.slr import CReg, For, If, Main, QReg, Return, SlrConverter, rad
from pecos.slr.ast import ast_to_guppy, slr_to_ast
from pecos.slr.ast.codegen.entry_wrapper import build_no_arg_entry_wrapper
from pecos.slr.ast.codegen.guppy import GuppyCodegenError
from pecos.slr.ast.nodes import (
    AssignOp,
    BinaryExpr,
    BinaryOp,
    BitExpr,
    BitRef,
    ForStmt,
    IfStmt,
    LiteralExpr,
    ReturnOp,
    UnaryExpr,
    UnaryOp,
    VarExpr,
)
from pecos.slr.cops import DIV
from pecos.slr.qeclib import qubit as qb
from pecos.slr.slr_converter import _load_and_compile_entry

BIT = BitExpr(ref=BitRef(register="d", index=0))
SYMBOLIC_BIT = VarExpr(name="d[i]")
LOOP = VarExpr(name="i")
ONE = LiteralExpr(value=1)
TWO = LiteralExpr(value=2)
FLOAT = LiteralExpr(value=1.5)
TRUE = LiteralExpr(value=True)
NUMPY_TRUE = LiteralExpr(value=np.bool_(True))
REGISTER = VarExpr(name="c")
OTHER_REGISTER = VarExpr(name="d")
OPERANDS = (BIT, SYMBOLIC_BIT, LOOP, ONE, TWO, FLOAT, TRUE, NUMPY_TRUE, REGISTER, OTHER_REGISTER)


def _binary(op: BinaryOp, left, right) -> BinaryExpr:
    return BinaryExpr(op=op, left=left, right=right)


def _source(conditions, assignments=()) -> tuple[object, str]:
    """Emit each condition as an `if`, inside a loop that defines the variable `i`."""
    program = slr_to_ast(Main(CReg("c", 4), CReg("d", 4)))
    target = BitRef(register="c", index=1)
    body = tuple(IfStmt(condition=condition, then_body=()) for condition in conditions)
    body += tuple(AssignOp(target=target, value=value) for value in assignments)
    loop = ForStmt(variable="i", start=LiteralExpr(value=0), stop=TWO, body=body)
    program = replace(program, body=(loop,))
    return program, ast_to_guppy(program)


def _compile(conditions, tmp_path, assignments=()) -> None:
    program, source = _source(conditions, assignments)
    path = tmp_path / "generated.py"
    path.write_text(source + build_no_arg_entry_wrapper(program)[0])
    _load_and_compile_entry(path, "_guppy_expression_types_generated")


def _accepted(condition=None, assignment=None) -> bool:
    try:
        _source([] if condition is None else [condition], [] if assignment is None else [assignment])
    except GuppyCodegenError:
        return False
    return True


def test_every_accepted_expression_compiles(tmp_path) -> None:
    binary = [_binary(op, *pair) for op, pair in product(BinaryOp, product(OPERANDS, repeat=2))]
    unary = [UnaryExpr(op=op, operand=operand) for op, operand in product(UnaryOp, OPERANDS)]
    accepted = [condition for condition in binary + unary if _accepted(condition)]
    assignments = [value for value in binary + unary if _accepted(assignment=value)]
    assert len(accepted) > 100
    assert len(assignments) > 50
    _compile(accepted, tmp_path, assignments)


@pytest.mark.parametrize(
    "condition",
    [
        _binary(BinaryOp.EQ, LOOP, ONE),
        _binary(BinaryOp.LT, LOOP, TWO),
        _binary(BinaryOp.LT, _binary(BinaryOp.ADD, LOOP, FLOAT), TWO),
        _binary(BinaryOp.XOR, LOOP, TWO),
        _binary(BinaryOp.LSHIFT, LOOP, ONE),
        _binary(BinaryOp.EQ, _binary(BinaryOp.DIV, LOOP, TWO), FLOAT),
        _binary(BinaryOp.XOR, BIT, ONE),
        _binary(BinaryOp.XOR, ONE, SYMBOLIC_BIT),
        _binary(BinaryOp.EQ, SYMBOLIC_BIT, ONE),
        _binary(BinaryOp.AND, BIT, _binary(BinaryOp.EQ, LOOP, ONE)),
        _binary(BinaryOp.EQ, REGISTER, LOOP),
        _binary(BinaryOp.EQ, TWO, REGISTER),
        _binary(BinaryOp.GE, REGISTER, OTHER_REGISTER),
        UnaryExpr(op=UnaryOp.NOT, operand=SYMBOLIC_BIT),
        UnaryExpr(op=UnaryOp.NOT, operand=LOOP),
        UnaryExpr(op=UnaryOp.NEG, operand=TRUE),
        _binary(BinaryOp.ADD, LiteralExpr(value=np.int64(1)), LiteralExpr(value=np.float64(1.5))),
        NUMPY_TRUE,
        _binary(BinaryOp.XOR, BIT, NUMPY_TRUE),
        LOOP,
        FLOAT,
    ],
)
def test_expressions_guppy_accepts_are_kept(condition) -> None:
    assert _accepted(condition)


@pytest.mark.parametrize(
    "value",
    [
        _binary(BinaryOp.XOR, LiteralExpr(value=0), ONE),
        _binary(BinaryOp.XOR, BIT, _binary(BinaryOp.XOR, LiteralExpr(value=0), ONE)),
        _binary(BinaryOp.XOR, BIT, LiteralExpr(value=np.int64(1))),
        LiteralExpr(value=np.int64(1)),
    ],
)
def test_bit_literals_stay_bits(value, tmp_path) -> None:
    _compile([], tmp_path, [value])


def test_float_division_stays_division() -> None:
    assert "(1.5 / 2)" in _source([_binary(BinaryOp.DIV, FLOAT, TWO)])[1]


@pytest.mark.parametrize(("numerator", "expected"), [(1, 0), (7, 1)])
def test_integer_division_truncates_like_the_qir_backend(numerator, expected) -> None:
    q, m = QReg("q", 1), CReg("m", 1)
    program = Main(q, m, If(DIV(numerator, 2)).Then(qb.X(q[0])), qb.MeasureZ(q[0]) > m[0], Return(m))
    package = SlrConverter(program).hugr()
    results = sim(Hugr(package.to_str().encode("utf-8"))).classical(selene_engine()).qubits(1).seed(42).run(1).to_dict()
    assert results == {"measurement_0": [expected]}


@pytest.mark.parametrize(
    "condition",
    [
        _binary(BinaryOp.XOR, BIT, FLOAT),
        _binary(BinaryOp.XOR, BIT, _binary(BinaryOp.ADD, BIT, ONE)),
        _binary(BinaryOp.XOR, BIT, LOOP),
        _binary(BinaryOp.LT, BIT, BIT),
        _binary(BinaryOp.ADD, BIT, ONE),
        _binary(BinaryOp.ADD, NUMPY_TRUE, TWO),
        _binary(BinaryOp.EQ, BIT, FLOAT),
        _binary(BinaryOp.LSHIFT, FLOAT, ONE),
        _binary(BinaryOp.ADD, REGISTER, ONE),
        _binary(BinaryOp.AND, REGISTER, OTHER_REGISTER),
        _binary(BinaryOp.EQ, REGISTER, BIT),
        _binary(BinaryOp.EQ, REGISTER, FLOAT),
        # Guppy accepts these, but SLR `&` and `|` are not its logical `and` and `or`.
        _binary(BinaryOp.AND, LOOP, TWO),
        _binary(BinaryOp.OR, LOOP, BIT),
        UnaryExpr(op=UnaryOp.NEG, operand=BIT),
        UnaryExpr(op=UnaryOp.NOT, operand=REGISTER),
        UnaryExpr(op=UnaryOp.NOT, operand=LiteralExpr(value=rad(0.5))),
    ],
)
def test_mismatched_operands_name_the_operator(condition) -> None:
    with pytest.raises(GuppyCodegenError, match="Guppy operator"):
        _source([condition])


def test_a_return_value_is_checked_like_a_condition() -> None:
    program = slr_to_ast(Main(CReg("c", 4), CReg("d", 4)))
    value = UnaryExpr(op=UnaryOp.NEG, operand=BIT)
    with pytest.raises(GuppyCodegenError, match="Guppy operator NEG"):
        ast_to_guppy(replace(program, body=(ReturnOp(values=(value,)),)))


@pytest.mark.parametrize("condition", [REGISTER, LiteralExpr(value=rad(0.5))])
def test_condition_must_be_a_boolean_or_number(condition) -> None:
    with pytest.raises(GuppyCodegenError, match="Unsupported condition"):
        _source([condition])


@pytest.mark.parametrize(
    ("op", "computed"),
    [
        (BinaryOp.LT, _binary(BinaryOp.LSHIFT, ONE, LiteralExpr(value=63))),
        (BinaryOp.GE, _binary(BinaryOp.SUB, LiteralExpr(value=0), ONE)),
        (BinaryOp.LE, LOOP),
        (BinaryOp.EQ, _binary(BinaryOp.DIV, _binary(BinaryOp.LSHIFT, ONE, LiteralExpr(value=63)), TWO)),
    ],
)
def test_register_comparison_rejects_signed_integers(op, computed) -> None:
    with pytest.raises(GuppyCodegenError, match="register comparison"):
        _source([_binary(op, REGISTER, computed)])


def test_register_comparison_does_not_use_a_name_a_program_can_shadow() -> None:
    shadow, c = CReg("int", 1), CReg("c", 1)
    SlrConverter(Main(shadow, c, If(c == 0).Then(c[0].set(1)))).hugr()


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        (2**63, "does not fit"),
        (-(2**63) - 1, "does not fit"),
        (float("inf"), "no literal"),
        (float("nan"), "no literal"),
    ],
)
def test_literals_guppy_cannot_write_are_rejected(value, reason) -> None:
    with pytest.raises(GuppyCodegenError, match=reason):
        _source([_binary(BinaryOp.LT, LiteralExpr(value=value), ONE)])


@pytest.mark.parametrize("value", [-1, 2**63])
def test_register_comparison_rejects_literals_outside_its_range(value) -> None:
    with pytest.raises(GuppyCodegenError, match="unsigned integers"):
        _source([_binary(BinaryOp.LT, REGISTER, LiteralExpr(value=value))])


def test_register_comparison_width_limit() -> None:
    wide, widest, d = CReg("wide", 63), CReg("widest", 64), CReg("d", 1)
    SlrConverter(Main(wide, d, If(wide == 1).Then(d[0].set(1)))).hugr()
    with pytest.raises(GuppyCodegenError, match="'widest' of width 64"):
        SlrConverter(Main(widest, d, If(widest == 1).Then(d[0].set(1)))).guppy()


@pytest.mark.parametrize("stop", [1.5, "register", "bit"])
def test_for_bounds_are_integers(stop) -> None:
    c = CReg("c", 4)
    bounds = {1.5: 1.5, "register": c, "bit": c[0]}
    with pytest.raises(GuppyCodegenError, match="For bound"):
        SlrConverter(Main(c, For("i", 0, bounds[stop]).Do(c[0].set(1)))).guppy()


@pytest.mark.parametrize(
    "compare",
    [
        lambda c, d: c == d,
        lambda c, d: c != d,
        lambda c, d: c < d,
        lambda c, d: c <= d,
        lambda c, d: c > d,
        lambda c, d: c >= d,
    ],
    ids=["eq", "ne", "lt", "le", "gt", "ge"],
)
@pytest.mark.parametrize(
    ("left_width", "right_width", "left", "right"),
    [(4, 4, 5, 5), (4, 4, 3, 12), (4, 4, 12, 3), (1, 4, 1, 12), (63, 4, 2**62, 3)],
)
def test_register_comparison_compares_unsigned_values(compare, left_width, right_width, left, right) -> None:
    c, d, q, m = CReg("c", left_width), CReg("d", right_width), QReg("q", 2), CReg("m", 2)
    program = Main(
        q,
        m,
        c,
        d,
        c.set(left),
        d.set(right),
        If(compare(c, d)).Then(qb.X(q[0])),
        If(compare(c, right)).Then(qb.X(q[1])),
        qb.MeasureZ(q) > m,
        Return(m),
    )
    package = SlrConverter(program).hugr()
    results = sim(Hugr(package.to_str().encode("utf-8"))).classical(selene_engine()).qubits(2).seed(42).run(1).to_dict()
    expected = int(compare(left, right))
    assert results == {"measurement_0": [expected], "measurement_1": [expected]}
