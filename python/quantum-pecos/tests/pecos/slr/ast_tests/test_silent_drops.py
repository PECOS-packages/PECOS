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

"""Code generators must implement operations or reject them explicitly (#1189)."""

from dataclasses import replace

import pytest
from pecos.slr import Barrier, Block, CReg, For, If, Main, Parallel, Permute, Print, QReg, Repeat, SlrConverter, While
from pecos.slr.ast import slr_to_ast
from pecos.slr.ast.codegen import generate
from pecos.slr.ast.nodes import (
    AllocatorDecl,
    BarrierOp,
    BlockCall,
    ForStmt,
    LiteralExpr,
    PermuteOp,
    RepeatStmt,
    Statement,
    VarExpr,
)
from pecos.slr.gen_codes.gen_qir import QIRGenerator
from pecos.slr.gen_codes.gen_quantum_circuit import QuantumCircuitGenerator
from pecos.slr.gen_codes.gen_stim import StimGenerator
from pecos.slr.qeclib import qubit as qb
from pecos_rslib import Qasm, sim, state_vector


@pytest.mark.parametrize("bounds", [(0, 3), (2, 5), (1, 7, 2), (5, -2, -2), (3, 3), (4, 1)])
def test_qasm_static_for_matches_unrolled_and_stim(bounds) -> None:
    """Compare exact QASM and deterministic outcomes using independent simulators."""
    q, c = QReg("q", 2), CReg("c", 2)
    body = (qb.X(q[0]), qb.CX(q[0], q[1]))
    program = Main(q, c, For("i", *bounds).Do(*body), qb.Measure(q) > c)
    explicit = Main(q, c, *(body * len(range(*bounds))), qb.Measure(q) > c)
    qasm = SlrConverter(program).qasm()
    assert qasm == SlrConverter(explicit).qasm()
    records = SlrConverter(program).stim().compile_sampler(seed=7).sample(shots=4)
    expected = [int(row[0]) + 2 * int(row[1]) for row in records]
    assert sim(Qasm.from_string(qasm)).quantum(state_vector()).run(4).to_dict()["c"] == expected


@pytest.mark.parametrize("which", ["start", "stop", "step"])
@pytest.mark.parametrize("value", [VarExpr(name="n"), LiteralExpr(value=1.5), LiteralExpr(value=True)])
@pytest.mark.parametrize("backend", ["qasm", "stim", "qir", "quantum_circuit"])
def test_for_rejects_noninteger_bounds(backend, which, value) -> None:
    """Every static unroller rejects dynamic, fractional and boolean bounds."""
    node = ForStmt(variable="i", start=LiteralExpr(value=0), stop=LiteralExpr(value=3))
    program = replace(slr_to_ast(Main(QReg("q", 1))), body=(replace(node, **{which: value}),))
    with pytest.raises(NotImplementedError, match=f"For loop {which} bound"):
        generate(program, backend)


@pytest.mark.parametrize("backend", ["qasm", "stim", "qir", "quantum_circuit"])
def test_for_zero_step(backend) -> None:
    """A zero step must fail even when the body and range are empty."""
    node = ForStmt(variable="i", start=LiteralExpr(value=0), stop=LiteralExpr(value=0), step=LiteralExpr(value=0))
    program = replace(slr_to_ast(Main(QReg("q", 1))), body=(node,))
    with pytest.raises(NotImplementedError, match="For loop step"):
        generate(program, backend)


def test_qasm_while_rejected() -> None:
    """Runtime loops must not turn into a comment."""
    q, c = QReg("q", 1), CReg("c", 1)
    with pytest.raises(
        NotImplementedError,
        match=r"While: QASM 2.0 has no runtime loops",
    ):
        SlrConverter(Main(q, c, While(c == 1).Do(qb.X(q[0])))).qasm()


def test_qasm_print_rejected_in_for() -> None:
    """Unrolling must dispatch previously unreachable unsupported statements."""
    c = CReg("c", 1)
    with pytest.raises(NotImplementedError, match="Print"):
        SlrConverter(Main(c, For("i", 0, 2).Do(Print(c)))).qasm()


@pytest.mark.parametrize("backend", ["qasm", "stim", "qir", "quantum_circuit"])
@pytest.mark.parametrize("empty_loop", [False, True])
def test_unresolved_block_call_rejected(backend, empty_loop) -> None:
    """An absent block declaration must never erase a call."""
    program = replace(slr_to_ast(Main(QReg("q", 1))), body=(BlockCall(callee="missing", arg_bindings=()),))
    if empty_loop:
        program = replace(program, body=(RepeatStmt(count=0, body=program.body),))
    with pytest.raises(ValueError, match=r"undefined block 'missing'"):
        generate(program, backend)


@pytest.mark.parametrize("count", [1.5, "n"])
def test_qir_noninteger_repeat_rejected(count) -> None:
    """The QIR Repeat guard used to skip non-integer bodies."""
    q = QReg("q", 1)
    program = slr_to_ast(Main(q, qb.X(q[0])))
    program = replace(program, body=(RepeatStmt(count=count, body=program.body),))
    with pytest.raises(NotImplementedError, match="Repeat"):
        generate(program, "qir")


@pytest.mark.parametrize("named", [False, True])
def test_qir_barrier_in_for(named) -> None:
    """Global barriers are real calls; repeated calls share one declaration."""
    q = QReg("q", 2)
    code = SlrConverter(Main(q, For("i", 0, 3).Do(Barrier(q) if named else Barrier()))).qir()
    assert code.count("call void @__quantum__qis__barrier2__body(") == 3


def test_qir_unknown_barrier_rejected() -> None:
    """A misspelled barrier allocator must not disappear."""
    program = replace(slr_to_ast(Main(QReg("q", 1))), body=(BarrierOp(allocators=("missing",)),))
    with pytest.raises(ValueError, match=r"Barrier.*missing"):
        generate(program, "qir")


def test_qir_child_allocator_barrier() -> None:
    """Barriers on child allocators resolve their slots rather than disappearing."""
    program = slr_to_ast(Main(QReg("q", 3)))
    program = replace(
        program,
        declarations=(
            *program.declarations,
            AllocatorDecl(name="padding", capacity=1, parent="q", offset=0),
            AllocatorDecl(name="child", capacity=2, parent="q", offset=1),
        ),
        body=(BarrierOp(allocators=("child",)),),
    )
    code = generate(program, "qir")
    assert "call void @__quantum__qis__barrier2__body(ptr inttoptr (i64 1 to ptr), ptr inttoptr (i64 2 to ptr))" in code


@pytest.mark.parametrize("backend", ["stim", "qir"])
def test_unknown_statement_rejected(backend) -> None:
    """Statement dispatch must fail closed for an operation without a lowering."""
    program = replace(slr_to_ast(Main(QReg("q", 1))), body=(Statement(),))
    with pytest.raises(NotImplementedError, match="Statement"):
        generate(program, backend)


@pytest.mark.parametrize("backend", ["stim", "quantum_circuit"])
@pytest.mark.parametrize(
    "region",
    ["top", "block", "parallel", "repeat", "empty_repeat", "for", "empty_for", "while", "then", "else"],
)
def test_deprecated_permute_rejected(backend, region) -> None:
    """Preflight all nested bodies, including unexecuted loops and else branches."""
    q, c = QReg("q", 2), CReg("c", 1)
    swap = Permute([q[0], q[1]], [q[1], q[0]])
    bodies = {
        "top": swap,
        "block": Block(swap),
        "parallel": Parallel(swap),
        "repeat": Repeat(2).block(swap),
        "empty_repeat": Repeat(0).block(swap),
        "for": For("i", 0, 2).Do(swap),
        "empty_for": For("i", 0, 0).Do(swap),
        "while": While(c == 1).Do(swap),
        "then": If(c == 1).Then(swap),
        "else": If(c == 1).Then(qb.X(q[0])).Else(swap),
    }
    generator = StimGenerator if backend == "stim" else QuantumCircuitGenerator
    with pytest.raises(NotImplementedError, match=rf"Permute.*SlrConverter.*\.{backend}\(\)"):
        generator(_internal=True).generate_block(Main(q, c, bodies[region]))


@pytest.mark.parametrize("backend", ["stim", "quantum_circuit"])
def test_ast_permute_compatibility(backend) -> None:
    """The supported AST path still relabels the same top-level program."""
    q = QReg("q", 2)
    program = Main(q, qb.X(q[0]), Permute([q[0], q[1]], [q[1], q[0]]), qb.Y(q[0]))
    explicit = Main(q, qb.X(q[0]), qb.Y(q[1]))
    assert str(getattr(SlrConverter(program), backend)()) == str(getattr(SlrConverter(explicit), backend)())


@pytest.mark.parametrize(
    ("sources", "targets"),
    [(("a", "b", "c"), ("b", "c", "a")), (("a[0]", "b"), ("b", "a[0]"))],
)
@pytest.mark.parametrize("classical", [False, True])
def test_qasm_unsupported_permute_shapes_rejected(sources, targets, classical) -> None:
    """Whole-register cycles and mixed reference shapes must not be dropped."""
    register = CReg if classical else QReg
    program = slr_to_ast(Main(*(register(name, 1) for name in ("a", "b", "c"))))
    program = replace(program, body=(PermuteOp(sources=sources, targets=targets),))
    with pytest.raises(NotImplementedError, match="Permute"):
        generate(program, "qasm")


def test_qasm_unequal_register_permute_rejected() -> None:
    """An unequal-sized quantum swap previously emitted only its comment."""
    a, b = QReg("a", 1), QReg("b", 2)
    with pytest.raises(NotImplementedError, match="Permute"):
        SlrConverter(Main(a, b, Permute(a, b))).qasm()


def test_qasm_for_permute_composes_each_iteration() -> None:
    """The newly reachable relabel must alternate the logical slot every time."""
    q = QReg("q", 2)
    program = Main(q, For("i", 0, 4).Do(Permute([q[0], q[1]], [q[1], q[0]]), qb.X(q[0])))
    code = SlrConverter(program).qasm()
    assert [line for line in code.splitlines() if line.startswith("x ")] == ["x q[1];", "x q[0];"] * 2


def test_qasm_for_classical_permute_declares_temporary_once() -> None:
    """Unrolling a swap must produce valid QASM with one temporary declaration."""
    c = CReg("c", 2)
    code = SlrConverter(Main(c, For("i", 0, 3).Do(Permute([c[0], c[1]], [c[1], c[0]])))).qasm()
    assert code.count("creg bitswap[1];") == 1
    assert code.count("bitswap[0] = c[0];") == 3


def test_qasm_for_permute_preparation_uses_current_mapping() -> None:
    """Every operation in an unrolled body must observe the current relabel."""
    q, c = QReg("q", 2), CReg("c", 2)
    program = Main(
        q,
        c,
        qb.X(q),
        For("i", 0, 3).Do(Permute([q[0], q[1]], [q[1], q[0]]), qb.PZ(q[0])),
        qb.Measure(q) > c,
    )
    qasm = SlrConverter(program).qasm()
    assert [line for line in qasm.splitlines() if line.startswith("reset ")] == [
        "reset q[1];",
        "reset q[0];",
        "reset q[1];",
    ]
    assert sim(Qasm.from_string(qasm)).quantum(state_vector()).run(4).to_dict()["c"] == [0] * 4
    assert not SlrConverter(program).stim().compile_sampler(seed=7).sample(shots=4).any()


def test_qasm_nested_for_preserves_if_context() -> None:
    """Unrolling in nested blocks must retain the enclosing runtime condition."""
    q, c = QReg("q", 1), CReg("c", 1)
    loop = For("i", 0, 2).Do(Repeat(2).block(For("j", 0, 2).Do(qb.X(q[0]))))
    program = Main(q, c, If(c == 1).Then(Block(loop)))
    explicit = Main(
        q,
        c,
        If(c == 1).Then(Repeat(2).block(qb.X(q[0]), qb.X(q[0])), Repeat(2).block(qb.X(q[0]), qb.X(q[0]))),
    )
    assert SlrConverter(program).qasm() == SlrConverter(explicit).qasm()


def test_deprecated_qir_rejects_partially_dropped_mixed_permute() -> None:
    """The legacy quantum branch used to ignore the classical half of a Permute."""
    q, c = QReg("q", 2), CReg("c", 2)
    program = Main(q, c, Permute([q[0], q[1], c[0], c[1]], [q[1], q[0], c[1], c[0]]))
    with pytest.raises(NotImplementedError, match=r"Permute.*SlrConverter.*\.qir\(\)"):
        QIRGenerator(_internal=True).generate_block(program)
    assert "; Permutation:" in SlrConverter(program).qir()


def test_deprecated_qir_rejects_unhandled_permute_operands() -> None:
    """A whole-register quantum/classical swap fell through every legacy branch."""
    q, c = QReg("q", 1), CReg("c", 1)
    with pytest.raises(NotImplementedError, match=r"Permute.*SlrConverter.*\.qir\(\)"):
        QIRGenerator(_internal=True).generate_block(Main(q, c, Permute(q, c)))
