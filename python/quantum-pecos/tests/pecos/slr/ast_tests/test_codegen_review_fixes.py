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

"""Regression coverage for the #1189 review and #1238."""

import re
from dataclasses import replace
from typing import ClassVar

import pytest
from pecos.slr import Barrier, Block, CReg, For, If, Main, Permute, QAlloc, QReg, Repeat, SlrConverter, While
from pecos.slr.ast import slr_to_ast
from pecos.slr.ast.codegen import generate
from pecos.slr.ast.codegen.guppy import GuppyCodegenError
from pecos.slr.ast.nodes import AssignOp, BitExpr, BitRef, PermuteOp
from pecos.slr.gen_codes.gen_qasm import QASMGenerator
from pecos.slr.gen_codes.gen_qir import QIRGenerator
from pecos.slr.qeclib import qubit as qb
from pecos.slr.vars import LoopVar
from pecos_rslib import Qasm, sim, state_vector


@pytest.mark.parametrize("use", ["assignment", "bit_read", "bit_write", "condition", "nested_body", "nested_bound"])
@pytest.mark.parametrize("stop", [0, 3])
def test_qasm_for_rejects_variable_references(use, stop) -> None:
    """Reject reads and writes even in nested or unreachable loop bodies."""
    q, c = QReg("q", 1), CReg("c", 3)
    i = LoopVar("i")
    bodies = {
        "assignment": c.set(c + i),
        "bit_read": c[0].set(c[i]),
        "bit_write": c[i].set(c[0]),
        "condition": If(c == i).Then(qb.X(q[0])),
        "nested_body": For("j", 0, 2).Do(c.set(c + i)),
        "nested_bound": For("j", 0, i).Do(qb.X(q[0])),
    }
    with pytest.raises(NotImplementedError, match=r"For.*variable.*'i'"):
        SlrConverter(Main(q, c, For(i, 0, stop).Do(bodies[use]))).qasm()


def test_qasm_for_rejects_ast_bit_index() -> None:
    """Direct AST references must be checked as well as SLR string expressions."""
    c = CReg("c", 3)
    program = slr_to_ast(Main(c, For("i", 0, 3).Do(c.set(0))))
    body = (AssignOp(target="c", value=BitExpr(ref=BitRef(register="c", index="i"))),)
    program = replace(program, body=(replace(program.body[0], body=body),))
    with pytest.raises(NotImplementedError, match=r"For.*variable.*'i'"):
        generate(program, "qasm")


def test_qasm_for_without_variable_still_unrolls() -> None:
    """Similar identifier names must not be mistaken for the loop variable."""
    q, bits = QReg("q", 1), CReg("bits", 3)
    body = (bits.set(bits + 1), qb.X(q[0]))
    assert (
        SlrConverter(Main(q, bits, For("i", 0, 3).Do(*body))).qasm()
        == SlrConverter(
            Main(q, bits, *(body * 3)),
        ).qasm()
    )


@pytest.mark.parametrize("backend", ["qasm", "stim", "qir", "quantum_circuit", "guppy"])
@pytest.mark.parametrize("whole", [False, True])
def test_permute_rejects_cross_kind_pairs(backend, whole) -> None:
    """A classical bit cannot become a qubit, including before later operations."""
    q, c = QReg("q", 2), CReg("c", 2)
    permute = Permute(q, c) if whole else Permute([q[0], c[0]], [c[0], q[0]])
    error = GuppyCodegenError if backend == "guppy" else NotImplementedError
    with pytest.raises(error, match="Permute"):
        getattr(SlrConverter(Main(q, c, permute, qb.X(q[0]), qb.Measure(q) > c)), backend)()


@pytest.mark.parametrize("classical_first", [False, True])
def test_qasm_rejects_mixed_element_collections(classical_first) -> None:
    """Even like-kind pairs cannot be dispatched solely from the first element."""
    q, c = QReg("q", 2), CReg("c", 2)
    a, b = (c, q) if classical_first else (q, c)
    with pytest.raises(NotImplementedError, match="Permute"):
        SlrConverter(Main(q, c, Permute([a[0], a[1], b[0], b[1]], [a[1], a[0], b[1], b[0]]))).qasm()


@pytest.mark.parametrize("backend", ["stim", "qir"])
@pytest.mark.parametrize("ref", ["missing[0]", "q[2]"])
def test_permute_rejects_undeclared_slots(backend, ref) -> None:
    """Indexed permutation refs must resolve to existing storage."""
    program = replace(slr_to_ast(Main(QReg("q", 2))), body=(PermuteOp(sources=(ref,), targets=(ref,)),))
    with pytest.raises(NotImplementedError, match="Permute"):
        generate(program, backend)


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("collision", [False, True])
@pytest.mark.parametrize("count", [1, 3])
def test_classical_permute_qasm_runs(legacy, collision, count) -> None:
    """Both emitters must produce legal, collision-free, singly declared temporaries."""
    q, c = QReg("q", 1), CReg("c", 2)
    reserved = [CReg("bitswap", 1), QReg("bitswapx", 1)] if collision else []
    program = Main(
        q,
        c,
        *reserved,
        c.set(1),
        Repeat(count).block(Permute([c[0], c[1]], [c[1], c[0]])),
        qb.X(q[0]),
    )
    if legacy:
        generator = QASMGenerator(_internal=True)
        generator.generate_block(program)
        code = generator.get_output()
    else:
        code = SlrConverter(program).qasm()
    result = sim(Qasm.from_string(code)).quantum(state_vector()).run(4).to_dict()
    assert result["c"] == [2] * 4
    temporary = "bitswapxx" if collision else "bitswap"
    assert code.count(f"creg {temporary}[1];") == 1


@pytest.mark.parametrize("kind", ["for", "empty_for", "while"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("permuting", [False, True])
def test_legacy_qir_rejects_loops(kind, nested, permuting) -> None:
    """Deprecated QIR must not treat a loop as a block executed once."""
    q, c = QReg("q", 1), CReg("c", 1)
    loop = {"for": For("i", 0, 3), "empty_for": For("i", 0, 0), "while": While(c == 1)}[kind].Do(qb.X(q[0]))
    if permuting:
        loop.Do(Permute([q[0]], [q[0]]))
    body = Block(Repeat(2).block(loop)) if nested else loop
    with pytest.raises(NotImplementedError, match=r"(For|While).*SlrConverter.*qir"):
        QIRGenerator(_internal=True).generate_block(Main(q, c, body))


@pytest.mark.parametrize("allocator", [False, True])
def test_barrier_qubits_widen_in_first_appearance_order(allocator) -> None:
    """Individual operands retain their owning allocator and do not duplicate it."""
    a, q = (QAlloc(2, name="a"), QAlloc(6, name="q")) if allocator else (QReg("a", 2), QReg("q", 6))
    program = Main(a, q, Barrier(q[1], a, q[3], a[0], q[5]))
    assert slr_to_ast(program).body[0].allocators == ("q", "a")
    assert "barrier q, a;" in SlrConverter(program).qasm()


def test_qasm_global_barrier_uses_only_declared_roots() -> None:
    base = QAlloc(5, name="base")
    child = base.child(2, name="data")
    code = SlrConverter(Main(base, child, Barrier())).qasm()
    assert "barrier base;" in code
    sim(Qasm.from_string(code)).quantum(state_vector()).run(1)


@pytest.mark.parametrize("child", [False, True])
@pytest.mark.parametrize("whole", [False, True])
def test_qasm_barrier_follows_permutation_like_qir(child, whole) -> None:
    """Project both backends' barrier operands to physical qubit indices."""
    if child:
        base = QAlloc(4, name="base")
        x, y = base.child(2, name="x"), base.child(2, name="y")
        declarations = (base, x, y)
        offsets = {"base": 0}
    else:
        x, y = QReg("x", 2), QReg("y", 2)
        declarations = (x, y)
        offsets = {"x": 0, "y": 2}
    permute = Permute(x, y) if whole else Permute([x[0], y[1]], [y[1], x[0]])
    if child and whole:
        permute = Permute([x[0], x[1], y[0], y[1]], [y[0], y[1], x[0], x[1]])
    converter = SlrConverter(Main(*declarations, permute, Barrier(x)))
    qasm = converter.qasm()
    barrier = next(line for line in qasm.splitlines() if line.startswith("barrier "))
    refs = re.findall(r"(\w+)\[(\d+)\]", barrier)
    actual = [offsets[name] + int(index) for name, index in refs]
    qir = converter.qir()
    call = next(line for line in qir.splitlines() if "call void @__quantum__qis__barrier" in line)
    # QIR represents physical qubit zero as null and all other indices as inttoptr.
    expected = [
        0 if match[1] else int(match[0])
        for match in re.findall(r"inttoptr \(i64 (\d+) to (?:ptr|%Qubit\*)\)|(null)", call)
    ]
    assert actual == expected == ([2, 3] if whole else [3, 1])


@pytest.mark.parametrize("nested", [False, True])
def test_barrier_block_inputs_resolve_before_widening(nested) -> None:
    """Single and bundled inputs keep their fences across both substitutions."""

    class Fence(Block):
        block_inputs: ClassVar[dict[str, str]] = {"a": "live_preserved", "data": "live_preserved"}

        def __init__(self, a, data) -> None:
            super().__init__()
            self.a, self.data = a, data
            fence = Barrier(a, data[0], data[1])
            self.extend(Repeat(2).block(Block(fence)) if nested else fence)

    q, r, spare = QReg("q", 4), QReg("r", 2), QReg("spare", 1)
    program = Main(q, r, spare, Fence(q[3], [q[1], r[0]]))
    ast = slr_to_ast(program)
    body = ast.block_decls[0].body[0]
    barrier = body.body[0] if nested else body
    assert barrier.allocators == ("a", "data")
    code = SlrConverter(program).qasm()
    assert [line for line in code.splitlines() if line.startswith("barrier ")] == ["barrier q, r;"] * (
        2 if nested else 1
    )
    assert "barrier6__body" in SlrConverter(program).qir()
