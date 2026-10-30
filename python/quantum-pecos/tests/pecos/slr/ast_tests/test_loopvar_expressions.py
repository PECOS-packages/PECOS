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

"""LoopVar comparisons build SLR expressions instead of Python bools (#1517)."""

import operator

import pytest
from pecos import Hugr, selene_engine, sim
from pecos.slr import CReg, For, If, Main, QReg, Return, SlrConverter
from pecos.slr.ast import slr_to_ast
from pecos.slr.ast.nodes import BinaryExpr, BinaryOp, ForStmt, IfStmt, LiteralExpr, VarExpr
from pecos.slr.cops import EQUIV, GE, GT, LE, LT, MINUS, NEQUIV, PLUS
from pecos.slr.gen_codes.gen_quantum_circuit import QuantumCircuitGenerator
from pecos.slr.qeclib import qubit as qb
from pecos.slr.vars import LoopVar


@pytest.mark.parametrize(
    ("build", "expected"),
    [
        (operator.eq, EQUIV),
        (operator.ne, NEQUIV),
        (operator.lt, LT),
        (operator.gt, GT),
        (operator.le, LE),
        (operator.ge, GE),
        (operator.add, PLUS),
        (operator.sub, MINUS),
    ],
)
def test_loopvar_operators_build_expressions(build, expected) -> None:
    i = LoopVar("i")
    expr = build(i, 1)
    assert type(expr) is expected
    assert expr.left is i
    assert expr.right == 1


def test_reflected_comparison_builds_expression() -> None:
    i = LoopVar("i")
    expr = operator.eq(0, i)  # int.__eq__ declines, so Python calls LoopVar.__eq__(i, 0)
    assert type(expr) is EQUIV
    assert expr.left is i


def test_loopvar_stays_hashable_by_identity() -> None:
    i, j = LoopVar("i"), LoopVar("i")
    values = {i: 0, j: 1}
    assert values[i] == 0
    assert values[j] == 1


def _condition_program(condition=lambda i: i == 0, stop=2):
    i = LoopVar("i")
    q, c = QReg("q", 1), CReg("c", 1)
    body = For(i, 0, stop).Do(If(condition(i)).Then(qb.X(q[0])))
    return Main(q, c, body, qb.Measure(q[0]) > c[0], Return(c))


def test_loopvar_condition_reaches_the_ast() -> None:
    program = slr_to_ast(_condition_program())
    loop = next(stmt for stmt in program.body if isinstance(stmt, ForStmt))
    (branch,) = loop.body
    assert isinstance(branch, IfStmt)
    assert branch.condition == BinaryExpr(op=BinaryOp.EQ, left=VarExpr(name="i"), right=LiteralExpr(value=0))


@pytest.mark.parametrize(
    ("condition", "measured"),
    [
        # Over i in range(3), X flips the qubit once per true iteration.
        (lambda i: i == 0, 1),
        (lambda i: i != 1, 0),
        (lambda i: i < 2, 0),
        (lambda i: i >= 1, 0),
        (lambda i: i > 1, 1),
        (lambda i: i - 1 == 0, 1),
    ],
)
def test_loopvar_condition_runs_in_guppy(condition, measured) -> None:
    # With the comparison folded to False the branch never ran and every shot measured 0.
    package = SlrConverter(_condition_program(condition, stop=3)).hugr()
    result = sim(Hugr(package.to_str().encode("utf-8"))).classical(selene_engine()).qubits(1).seed(7).run(8)
    raw = result.to_dict() if hasattr(result, "to_dict") else result
    assert raw == {"measurement_0": [measured] * 8}


@pytest.mark.parametrize(
    "build",
    [
        operator.and_,
        operator.or_,
        operator.xor,
        operator.lshift,
        operator.rshift,
    ],
)
def test_loopvar_bitwise_operators_are_unsupported(build) -> None:
    # Code generators treat bitwise operands as bools, which a loop variable is not.
    with pytest.raises(TypeError):
        build(LoopVar("i"), 1)


@pytest.mark.parametrize("build", [operator.invert, operator.neg])
def test_loopvar_unary_operators_are_unsupported(build) -> None:
    with pytest.raises(TypeError):
        build(LoopVar("i"))


def test_loopvar_outside_its_loop_is_rejected() -> None:
    i = LoopVar("i")
    q, c = QReg("q", 1), CReg("c", 1)
    program = Main(q, c, For(i, 0, 2).Do(qb.X(q[0])), If(i == 0).Then(qb.X(q[0])))
    with pytest.raises(ValueError, match=r"LoopVar 'i' is used outside the For loop that binds it"):
        slr_to_ast(program)


def test_qir_string_loop_label_may_name_a_declared_register() -> None:
    q, i = QReg("q", 1), CReg("i", 2)
    body = i.set(i + 1)
    looped = Main(q, i, For("i", 0, 2).Do(body))
    assert SlrConverter(looped).qir() == SlrConverter(Main(q, i, body, body)).qir()


@pytest.mark.parametrize(
    ("backend", "message"),
    [
        ("qasm", r"QASM codegen: For body references loop variable 'i'; binding it is not supported\."),
        ("qir", r"QIR codegen: For body references loop variable 'i'; binding it is not supported\."),
        ("stim", r"Stim codegen: unsupported If condition \(BinaryExpr\)"),
        ("quantum_circuit", r"If cannot be converted to QuantumCircuit format"),
    ],
)
def test_unrolling_backends_reject_loopvar_conditions(backend, message) -> None:
    with pytest.raises(NotImplementedError, match=message):
        getattr(SlrConverter(_condition_program()), backend)()


def test_deprecated_quantum_circuit_generator_still_unrolls_loopvar_indices() -> None:
    i = LoopVar("i")
    q = QReg("q", 2)
    generator = QuantumCircuitGenerator(_internal=True)
    generator.generate_block(Main(q, For(i, 0, 2).Do(qb.X(q[i]))))
    gates = [(symbol, sorted(locations)) for symbol, locations, _ in generator.get_circuit().items()]
    assert gates == [("X", [0]), ("X", [1])]
