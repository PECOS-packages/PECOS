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

"""A literal on the left of a classical register or bit operator builds an SLR expression."""

import operator

import pytest
from pecos.slr import CReg, Main, QReg, SlrConverter
from pecos.slr.cops import AND, LSHIFT, MINUS, OR, PLUS, RSHIFT, XOR
from pecos.slr.qeclib import qubit as qb
from pecos.slr.vars import LoopVar
from pecos_rslib import Qasm, sim, state_vector

REFLECTED = [
    (operator.add, PLUS),
    (operator.sub, MINUS),
    (operator.and_, AND),
    (operator.or_, OR),
    (operator.xor, XOR),
    (operator.lshift, LSHIFT),
    (operator.rshift, RSHIFT),
]


@pytest.mark.parametrize(("build", "expected"), REFLECTED)
@pytest.mark.parametrize("operand", ["register", "bit", "expression"])
def test_literal_left_operand_keeps_operand_order(build, expected, operand) -> None:
    c = CReg("c", 4)
    value = {"register": c, "bit": c[0], "expression": c + 1}[operand]
    expr = build(3, value)
    assert type(expr) is expected
    assert expr.left == 3
    assert expr.right is value


@pytest.mark.parametrize(
    ("build", "expected"),
    [
        # With a == 1, swapping the operands of -, >> or << changes the result.
        (lambda a: 3 - a, 2),
        (lambda a: 8 >> a, 4),
        (lambda a: 4 << a, 8),
        (lambda a: 3 + a, 4),
        (lambda a: 6 | a, 7),
    ],
)
def test_literal_left_operand_runs_in_qasm(build, expected) -> None:
    q, a, b = QReg("q", 1), CReg("a", 4), CReg("b", 4)
    code = SlrConverter(Main(q, a, b, a.set(1), b.set(build(a)))).qasm()
    result = sim(Qasm.from_string(code)).quantum(state_vector()).seed(7).run(1).to_dict()
    assert result["b"] == [expected]


NON_LITERAL_CASES = [
    *[(build, operand) for build, _ in REFLECTED for operand in ("register", "bit", "expression")],
    # LoopVar has only + and - (bitwise and shift operators are absent altogether).
    (operator.add, "loop variable"),
    (operator.sub, "loop variable"),
]


@pytest.mark.parametrize(("build", "operand"), NON_LITERAL_CASES)
def test_non_literal_left_operand_is_rejected(build, operand) -> None:
    c = CReg("c", 4)
    value = {"register": c, "bit": c[0], "expression": c + 1, "loop variable": LoopVar("i")}[operand]
    with pytest.raises(TypeError):
        build(object(), value)


@pytest.mark.parametrize(
    "build",
    [
        # A mistyped measurement redirect would otherwise become an expression the converter drops.
        lambda: qb.Measure(QReg("q", 1)[0]) >> CReg("c", 1),
        # Bitwise operators on a LoopVar stay unsupported from either side.
        lambda: LoopVar("i") & CReg("c", 1)[0],
    ],
)
def test_slr_object_left_operand_is_rejected(build) -> None:
    with pytest.raises(TypeError):
        build()
