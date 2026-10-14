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

"""If/else lowering to OpenQASM 2.0, checked by running the emitted QASM.

OpenQASM 2.0 guards one operation per `if` and has no else or nesting, so the
generators guard every branch operation separately, run an else-branch under the
negated comparison, and reject what that cannot express.
"""

import operator
from collections.abc import Callable

import pytest
from pecos.slr import CReg, If, Main, QReg, Repeat, SlrConverter
from pecos.slr.gen_codes.gen_qasm import QASMGenerator
from pecos.slr.qeclib import qubit as qb
from pecos.slr.qeclib.qubit.measures import Measure
from pecos_rslib import Qasm, sim, state_vector

SHOTS = 32


def _ast_qasm(program: Main) -> str:
    return SlrConverter(program).qasm()


def _legacy_qasm(program: Main) -> str:
    generator = QASMGenerator(_internal=True)
    generator.generate_block(program)
    return generator.get_output()


def _run(qasm: str) -> dict[str, list[int]]:
    return sim(Qasm.from_string(qasm)).quantum(state_vector()).seed(7).run(SHOTS).to_dict()


def _random_flag(q: QReg, f: CReg) -> tuple:
    """Put a uniformly random bit in f[0]."""
    return (qb.H(q[0]), Measure(q[0]) > f[0])


COMPARISONS = [operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge]


@pytest.mark.parametrize("compare", COMPARISONS, ids=lambda op: op.__name__)
@pytest.mark.parametrize("value", [0, 1])
def test_else_branch_runs_exactly_when_the_condition_is_false(compare: Callable, value: int) -> None:
    """c[0] marks the then-branch and c[1] the else-branch; exactly the right one runs."""
    q, f, c = QReg("q", 3), CReg("f", 1), CReg("c", 2)
    program = Main(
        q,
        f,
        c,
        *_random_flag(q, f),
        If(compare(f, value)).Then(qb.X(q[1])).Else(qb.X(q[2])),
        Measure(q[1]) > c[0],
        Measure(q[2]) > c[1],
    )
    results = _run(_ast_qasm(program))

    assert set(results["f"]) == {0, 1}
    for flag, record in zip(results["f"], results["c"], strict=True):
        taken = compare(flag, value)
        assert record == (0b01 if taken else 0b10), (flag, record)


def test_else_branch_on_a_bit_condition() -> None:
    """The issue reproducer: X on both branches flips q[1] on every shot."""
    q, f, c = QReg("q", 2), CReg("f", 1), CReg("c", 1)
    program = Main(
        q,
        f,
        c,
        *_random_flag(q, f),
        If(f[0] == 1).Then(qb.X(q[1])).Else(qb.X(q[1])),
        Measure(q[1]) > c[0],
    )
    results = _run(_ast_qasm(program))

    assert set(results["f"]) == {0, 1}
    assert results["c"] == [1] * SHOTS


def test_else_needs_a_comparison_condition() -> None:
    q, f = QReg("q", 1), CReg("f", 1)
    program = Main(q, f, If(f[0]).Then(qb.X(q[0])).Else(qb.Z(q[0])))
    with pytest.raises(NotImplementedError, match="comparison condition"):
        _ast_qasm(program)


# The legacy generator rejects every else block before nesting is reached.
@pytest.mark.parametrize(
    ("generate", "where"),
    [(_ast_qasm, "then"), (_ast_qasm, "else"), (_ast_qasm, "repeat"), (_legacy_qasm, "then"), (_legacy_qasm, "repeat")],
    ids=["ast-then", "ast-else", "ast-repeat", "legacy-then", "legacy-repeat"],
)
def test_nested_if_is_rejected(generate: Callable, where: str) -> None:
    q, f, g = QReg("q", 1), CReg("f", 1), CReg("g", 1)
    inner = If(g[0] == 1).Then(qb.X(q[0]))
    if where == "then":
        outer = If(f[0] == 1).Then(inner)
    elif where == "else":
        outer = If(f[0] == 1).Then(qb.Z(q[0])).Else(inner)
    else:
        outer = If(f[0] == 1).Then(Repeat(2).block(inner))
    with pytest.raises(NotImplementedError, match=r"[Nn]ested If"):
        generate(Main(q, f, g, outer))


def test_legacy_generator_rejects_else() -> None:
    q, f = QReg("q", 1), CReg("f", 1)
    program = Main(q, f, If(f[0] == 1).Then(qb.X(q[0])).Else(qb.Z(q[0])))
    with pytest.raises(NotImplementedError, match="else block"):
        _legacy_qasm(program)


def _set_flag(q: QReg, f: CReg) -> tuple:
    """Make f[0] = 1 on every shot."""
    return (qb.X(q[0]), Measure(q[0]) > f[0])


@pytest.mark.parametrize("generate", [_ast_qasm, _legacy_qasm], ids=["ast", "legacy"])
def test_writing_the_condition_last_is_allowed(generate: Callable) -> None:
    """Re-measuring into the condition bit as the final operation keeps the branch whole."""
    q, f, c = QReg("q", 3), CReg("f", 1), CReg("c", 1)
    program = Main(
        q,
        f,
        c,
        *_set_flag(q, f),
        If(f[0] == 1).Then(qb.X(q[1]), Measure(q[2]) > f[0]),
        Measure(q[1]) > c[0],
    )
    results = _run(generate(program))

    assert results["c"] == [1] * SHOTS
    assert results["f"] == [0] * SHOTS


@pytest.mark.parametrize("generate", [_ast_qasm, _legacy_qasm], ids=["ast", "legacy"])
def test_writing_another_bit_of_the_condition_register_is_allowed(generate: Callable) -> None:
    q, f, c = QReg("q", 3), CReg("f", 2), CReg("c", 1)
    program = Main(
        q,
        f,
        c,
        *_set_flag(q, f),
        If(f[0] == 1).Then(Measure(q[2]) > f[1], qb.X(q[1])),
        Measure(q[1]) > c[0],
    )
    assert _run(generate(program))["c"] == [1] * SHOTS


# Ifs that write a bit their condition reads and then guard another operation.
WRITE_THEN_MORE: dict[str, Callable[[QReg, CReg], If]] = {
    "measure-into-condition-bit": lambda q, f: If(f[0] == 1).Then(Measure(q[2]) > f[0], qb.X(q[1])),
    "measure-into-condition-register": lambda q, f: If(f == 1).Then(Measure(q[2]) > f[1], qb.X(q[1])),
    "assign-condition-bit": lambda q, f: If(f[0] == 1).Then(f[0].set(0), qb.X(q[1])),
    "assign-condition-register": lambda q, f: If(f == 1).Then(f.set(0), qb.X(q[1])),
    "assign-register-of-condition-bit": lambda q, f: If(f[0] == 1).Then(f.set(0), qb.X(q[1])),
    "multi-measure-writes-first": lambda q, f: If(f[0] == 1).Then(Measure(q[1], q[2]) > (f[0], f[1])),
}


@pytest.mark.parametrize("generate", [_ast_qasm, _legacy_qasm], ids=["ast", "legacy"])
@pytest.mark.parametrize("build", WRITE_THEN_MORE.values(), ids=WRITE_THEN_MORE.keys())
def test_writing_the_condition_before_another_guarded_operation_is_rejected(
    generate: Callable,
    build: Callable[[QReg, CReg], If],
) -> None:
    q, f = QReg("q", 3), CReg("f", 2)
    program = Main(q, f, build(q, f))
    with pytest.raises(NotImplementedError, match="writes a bit its condition reads"):
        generate(program)


def test_then_branch_writing_the_condition_last_runs_after_the_else_branch() -> None:
    """The branches are exclusive, so the else-branch is emitted first and reads the condition unchanged.

    The then-branch clears f[0] (measuring a fresh qubit); emitted then-first, the
    else-branch would see f[0] == 0 and also run.
    """
    q, f, c = QReg("q", 3), CReg("f", 1), CReg("c", 1)
    program = Main(
        q,
        f,
        c,
        *_random_flag(q, f),
        If(f[0] == 1).Then(Measure(q[2]) > f[0]).Else(qb.X(q[1])),
        Measure(q[1]) > c[0],
    )
    qasm = _ast_qasm(program)
    assert qasm.index("if(f[0] != 1) x q[1];") < qasm.index("if(f[0] == 1) measure q[2] -> f[0];")
    results = _run(qasm)

    assert results["f"] == [0] * SHOTS
    # X ran only on shots where the flag was 0, about half of them
    assert 0 < sum(results["c"]) < SHOTS


def test_both_branches_writing_the_condition_is_rejected() -> None:
    q, f = QReg("q", 2), CReg("f", 1)
    program = Main(q, f, If(f[0] == 1).Then(Measure(q[0]) > f[0]).Else(Measure(q[1]) > f[0]))
    with pytest.raises(NotImplementedError, match="both branches"):
        _ast_qasm(program)


@pytest.mark.parametrize("generate", [_ast_qasm, _legacy_qasm], ids=["ast", "legacy"])
def test_a_condition_write_does_not_carry_into_the_next_if(generate: Callable) -> None:
    """The first If ends by writing f[0]; the second If reads it afresh and runs X."""
    q, f, c = QReg("q", 3), CReg("f", 1), CReg("c", 1)
    program = Main(
        q,
        f,
        c,
        *_set_flag(q, f),
        If(f[0] == 1).Then(Measure(q[2]) > f[0]),
        If(f[0] == 0).Then(qb.X(q[1])),
        Measure(q[1]) > c[0],
    )
    assert _run(generate(program))["c"] == [1] * SHOTS
