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

"""Regressions from the fourth #1502 review."""

import re
from typing import ClassVar

import pytest
from pecos.slr import Barrier, Block, CReg, For, If, Main, Permute, QAlloc, QReg, Repeat, SlrConverter
from pecos.slr.ast import slr_to_ast
from pecos.slr.ast.codegen import generate
from pecos.slr.ast.optimizations import optimize
from pecos.slr.qeclib import qubit as qb
from pecos.slr.transforms.parallel_optimizer import ParallelOptimizer
from pecos.slr.vars import LoopVar


@pytest.mark.parametrize("kind", ["QReg", "CReg", "QAlloc"])
@pytest.mark.parametrize(
    "scope",
    ["main", "nested", "else", "loop", "block_in_loop", "sibling_before", "sibling_after"],
)
def test_loopvar_register_collision_rejected(kind, scope) -> None:
    i, c = LoopVar("i"), CReg("c", 2)
    register = QAlloc(2, name="i") if kind == "QAlloc" else {"QReg": QReg, "CReg": CReg}[kind]("i", 2)
    loop = For(i, 0, 2).Do(c.set(i))
    if scope == "main":
        program = Main(c, register, Block(loop))
    elif scope == "nested":
        program = Main(c, Block(register, loop))
    elif scope == "else":
        program = Main(c, If(c == 0).Then().Else(Block(register, loop)))
    elif scope == "loop":
        loop.Do(register)
        program = Main(c, loop)
    elif scope == "block_in_loop":
        loop.Do(Block(register))
        program = Main(c, loop)
    elif scope == "sibling_before":
        program = Main(c, Block(register), loop)
    else:
        program = Main(c, loop, Block(register))
    with pytest.raises(ValueError, match=rf"For.*loop variable 'i'.*{kind}.*'i'.*[Rr]ename"):
        slr_to_ast(program)


@pytest.mark.parametrize("bound", ["register", "qubit"])
def test_loopvar_block_input_formal_collision_rejected(bound) -> None:
    quantum, c, i = QReg("quantum", 2), CReg("c", 2), LoopVar("i")

    class UsesQuantum(Block):
        block_inputs: ClassVar[dict[str, str]] = {"i": "live_preserved"}

        def __init__(self) -> None:
            super().__init__()
            self.i = quantum if bound == "register" else quantum[0]
            self.extend(For(i, 0, 2).Do(c.set(i)))

    with pytest.raises(ValueError, match=r"For loop variable 'i' conflicts with block input 'i'\. Rename"):
        slr_to_ast(Main(quantum, c, UsesQuantum()))


@pytest.mark.parametrize("transformed", [False, True])
def test_string_loop_variable_collision_still_unrolls(transformed) -> None:
    c, i = CReg("c", 2), CReg("i", 2)
    body = c.set(i)
    program = Main(c, i, For("i", 0, 2).Do(body))
    if transformed:
        program = ParallelOptimizer().transform(program)
    assert SlrConverter(program).qasm() == SlrConverter(Main(c, i, body, body)).qasm()


@pytest.mark.parametrize("repeated", [False, True])
def test_stim_symbolic_write_invalidates_register(repeated) -> None:
    q, d, i = QReg("q", 2), CReg("d", 2), LoopVar("i")
    loop = For(i, 0, 2).Do(d[i].set(0))
    writes = Repeat(2).block(loop) if repeated else loop

    def program(*assignments):
        return Main(
            q,
            d,
            qb.X(q[0]),
            qb.Measure(q[0]) > d[0],
            *assignments,
            If(d[0] == 1).Then(qb.X(q[1])),
            qb.Measure(q[1]) > d[1],
        )

    with pytest.raises(NotImplementedError, match="no longer comes from a tracked measurement") as explicit:
        SlrConverter(program(d[0].set(0), d[1].set(0))).stim()
    with pytest.raises(NotImplementedError) as symbolic:
        SlrConverter(program(writes)).stim()
    assert str(symbolic.value) == str(explicit.value)


@pytest.mark.parametrize("classical", [False, True])
@pytest.mark.parametrize("optimized", [False, True])
def test_qasm_empty_main_barrier_is_empty(classical, optimized) -> None:
    declarations = (CReg("c", 1),) if classical else ()
    program = slr_to_ast(Main(*declarations, Barrier()))
    if optimized:
        program = optimize(program, level=3).program
    assert generate(program, "qasm") == SlrConverter(Main(*declarations)).qasm()


@pytest.mark.parametrize("optimized", [False, True])
def test_qasm_empty_fragment_barrier_still_rejected(optimized) -> None:
    program = slr_to_ast(Block(Barrier(), block_name="Main"))
    if optimized:
        program = optimize(program, level=3).program
    with pytest.raises(NotImplementedError, match="global Barrier with no declared registers"):
        generate(program, "qasm")


@pytest.mark.parametrize("backend", ["qasm", "stim", "qir", "quantum_circuit"])
@pytest.mark.parametrize("classical", [False, True])
def test_symbolic_permute_index_has_actionable_error(backend, classical) -> None:
    q, c, i = QReg("q", 2), CReg("c", 2), LoopVar("i")
    reg = c if classical else q
    program = Main(q, c, For(i, 0, 2).Do(Permute([reg[i], reg[0]], [reg[0], reg[i]])))
    with pytest.raises(NotImplementedError, match=rf"Permute.*{reg.sym}\[i\].*loop-variable indices.*not supported"):
        getattr(SlrConverter(program), backend)()


def test_stim_local_classical_permute_matches_main_scope() -> None:
    q, loc = QReg("q", 2), CReg("loc", 2)
    program = Main(q, Block(loc, qb.Measure(q) > loc, Permute([loc[0], loc[1]], [loc[1], loc[0]])))
    outer = Main(q, loc, qb.Measure(q) > loc, Permute([loc[0], loc[1]], [loc[1], loc[0]]))
    assert str(SlrConverter(program).stim()) == str(SlrConverter(outer).stim())


@pytest.mark.parametrize("parent_first", [False, True])
@pytest.mark.parametrize("offset", [False, True])
def test_overlapping_barriers_fence_each_physical_qubit_once(parent_first, offset) -> None:
    q = QAlloc(4, name="q")
    declarations = [q]
    if offset:
        declarations.append(q.child(2, name="spare"))
    d = q.child(2, name="d")
    declarations.append(d)
    operands = (q, d[0]) if parent_first else (d[0], q)
    converter = SlrConverter(Main(*declarations, Barrier(*operands)))
    barrier = next(line for line in converter.qasm().splitlines() if line.startswith("barrier "))
    qasm_qubits = []
    for ref in barrier.removeprefix("barrier ").removesuffix(";").split(", "):
        if ref == "q":
            qasm_qubits.extend(range(4))
        else:
            qasm_qubits.append(int(re.fullmatch(r"q\[(\d+)\]", ref)[1]))
    call = next(line for line in converter.qir().splitlines() if "call void @__quantum__qis__barrier" in line)
    qir_qubits = [
        0 if null else int(index)
        for index, null in re.findall(r"inttoptr \(i64 (\d+) to (?:ptr|%Qubit\*)\)|(null)", call)
    ]
    expected = [2, 3, 0, 1] if offset and not parent_first else [0, 1, 2, 3]
    assert qasm_qubits == expected
    assert qir_qubits == expected


def test_bracketed_register_name_is_a_whole_register_target() -> None:
    q, a = QReg("q", 2), CReg("a[0]", 2)
    assert SlrConverter(Main(q, a, a.set(1))).qir()
    stale = Main(q, a, qb.Measure(q[0]) > a[0], a.set(0), If(a[0] == 1).Then(qb.X(q[1])))
    with pytest.raises(NotImplementedError, match="no longer comes from a tracked measurement"):
        SlrConverter(stale).stim()
