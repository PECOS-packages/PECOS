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

"""Compatibility regressions from the second #1189 review."""

from typing import ClassVar

import pytest
from pecos.slr import Barrier, Block, Comment, CReg, For, If, Main, Permute, QAlloc, QReg, Repeat, SlrConverter
from pecos.slr.ast import slr_to_ast
from pecos.slr.ast.codegen.guppy import GuppyCodegenError
from pecos.slr.ast.codegen.qir import AstToQir
from pecos.slr.qeclib import qubit as qb
from pecos.slr.qeclib.generic.check import Check
from pecos.slr.vars import LoopVar
from pecos_rslib import Qasm, sim, state_vector


@pytest.mark.parametrize("form", ["tuple", "slice", "nested"])
@pytest.mark.parametrize("allocator", [False, True])
def test_barrier_collections(form, allocator) -> None:
    """Collections expand in first-appearance order without register reprs."""
    q, a = (QAlloc(3, name="q"), QAlloc(2, name="a")) if allocator else (QReg("q", 3), QReg("a", 2))
    operands = {"tuple": (q, a), "slice": [q[0], q[1]] if allocator else q[0:2], "nested": ([q[1], (a,)], q)}
    program = Main(q, a, Barrier(operands[form]))
    names = ("q",) if form == "slice" else ("q", "a")
    assert slr_to_ast(program).body[0].allocators == names
    code = SlrConverter(program).qasm()
    assert f"barrier {', '.join(names)};" in code
    sim(Qasm.from_string(code)).quantum(state_vector()).run(1)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("operand", [CReg("bad", 2), "bad", 42])
def test_barrier_rejects_nonquantum_operands(operand, nested) -> None:
    value = ([operand],) if nested else operand
    with pytest.raises(ValueError, match=r"Barrier.*(bad|42)"):
        slr_to_ast(Main(Barrier(value)))


@pytest.mark.parametrize("nested", [False, True])
def test_barrier_symbolic_qubit_uses_gate_validation(nested) -> None:
    q, i = QReg("q", 2), LoopVar("i")
    value = ([q[i]],) if nested else q[i]
    with pytest.raises(ValueError, match=r"Gate Barrier: loop-variable qubit indices.*q\[i\]"):
        SlrConverter(Main(q, For(i, 0, 2).Do(Barrier(value)))).qasm()


def test_qasm_global_barrier_fragment_rejected() -> None:
    q = QReg("q", 2)
    with pytest.raises(NotImplementedError, match="global Barrier with no declared registers"):
        SlrConverter(Block(qb.H(q[0]), Barrier(), qb.H(q[1]))).qasm()


@pytest.mark.parametrize("shape", ["qubit", "allocator", "assignment", "condition"])
def test_qasm_loop_name_can_name_a_declared_register(shape) -> None:
    q, i = QReg("q", 1), CReg("i", 2)
    if shape in ("qubit", "allocator"):
        i = QReg("i", 1) if shape == "qubit" else QAlloc(1, name="i")
        body = qb.X(i[0])
    elif shape == "assignment":
        body = i.set(i + 1)
    else:
        body = If(i == 1).Then(qb.X(q[0]))
    looped = Main(q, i, For("i", 0, 2).Do(body))
    explicit = Main(q, i, body, body)
    assert SlrConverter(looped).qasm() == SlrConverter(explicit).qasm()


def scratch_caller_program(fence):
    q, c = QReg("q", 3), CReg("c", 1)
    return Main(q, c, qb.PZ(q), fence(q[0], q[1]), Check([q[0], q[1]], "ZZ", q[2], c[0]))


class ScratchFence(Block):
    """A fence around the lifecycle does not consume scratch state."""

    block_inputs: ClassVar[dict[str, str]] = {"a": "scratch", "out": "live_preserved"}

    def __init__(self, a, out, fence=Barrier) -> None:
        super().__init__()
        self.a, self.out = a, out
        self.extend(fence(a), qb.PZ(a), qb.Measure(a) > out, fence(a))


def scratch_internal_program(fence):
    q, c = QReg("q", 1), CReg("c", 1)
    return Main(q, c, ScratchFence(q[0], c[0], fence))


@pytest.mark.parametrize("build", [scratch_caller_program, scratch_internal_program])
def test_scratch_barriers_are_comments(build) -> None:
    """Both validators must ignore barriers, as the Guppy emitter does."""
    actual = SlrConverter(build(Barrier)).guppy()
    expected = SlrConverter(build(lambda *_args: Comment("barrier"))).guppy()
    assert actual == expected
    assert "# barrier" in actual


@pytest.mark.parametrize("repeat", [False, True])
def test_stim_classical_bit_permute_restores_circuit_and_invalidates_records(repeat) -> None:
    q, c = QReg("q", 2), CReg("c", 2)
    swap = Permute([c[0], c[1]], [c[1], c[0]])
    operation = Repeat(2).block(swap) if repeat else swap
    measurements = (qb.Measure(q[1]) > c[1], qb.Measure(q[0]) > c[0])
    prefix = (q, c, qb.X(q[0]), *measurements, operation)
    converter = SlrConverter(Main(*prefix, *measurements))
    assert str(converter.stim()) == "X 0\nM 1 0 1 0"
    circuit, records = converter.stim_with_records()
    assert circuit == converter.stim()
    assert records == (("c", 1), ("c", 0)) * 2
    with pytest.raises(NotImplementedError, match="no longer comes from a tracked measurement"):
        SlrConverter(Main(*prefix, If(c[0] == 1).Then(qb.X(q[1])))).stim()


@pytest.mark.parametrize("cross_kind", [False, True])
def test_stim_mixed_permute_remains_rejected(cross_kind) -> None:
    q, c = QReg("q", 2), CReg("c", 2)
    swap = (
        Permute([q[0], c[0]], [c[0], q[0]])
        if cross_kind
        else Permute(
            [q[0], q[1], c[0], c[1]],
            [q[1], q[0], c[1], c[0]],
        )
    )
    with pytest.raises(NotImplementedError, match="Permute"):
        SlrConverter(Main(q, c, swap, qb.X(q[0]))).stim()


@pytest.mark.parametrize("whole", [False, True])
def test_qir_classical_permute_still_supported(whole) -> None:
    c, d = CReg("c", 1), CReg("d", 1)
    swap = Permute(c, d) if whole else Permute([c[0], d[0]], [d[0], c[0]])
    generator = AstToQir()
    generator.generate(slr_to_ast(Main(c, d, swap, c[0].set(1))))
    assert generator.context.permutation_map == {("c", 0): ("d", 0), ("d", 0): ("c", 0)}


def test_barrier_collections_preserve_block_bindings() -> None:
    """Recursive collection expansion must retain whole and single input bindings."""

    class Fence(Block):
        block_inputs: ClassVar[dict[str, str]] = {"whole": "live_preserved", "single": "live_preserved"}

        def __init__(self, whole, single) -> None:
            super().__init__()
            self.whole, self.single = whole, single
            self.extend(Barrier((whole, [single, (whole[1],)])))

    q, a = QReg("q", 2), QReg("a", 1)
    program = Main(q, a, Fence(q, a[0]))
    assert slr_to_ast(program).block_decls[0].body[0].allocators == ("whole", "single")
    assert "barrier q, a;" in SlrConverter(program).qasm()


def test_symbolic_classical_measurement_result_rejected() -> None:
    """The shared bit-ref converter must reject symbolic measurement destinations too."""
    q, c, i = QReg("q", 1), CReg("c", 2), LoopVar("i")
    with pytest.raises(NotImplementedError, match=r"For loop variable 'i'.*symbolic bit indices"):
        slr_to_ast(Main(q, c, For(i, 0, 2).Do(qb.Measure(q[0]) > c[i])))


def symbolic_classical_assignment_program() -> Main:
    i = LoopVar("i")
    q, c, d = QReg("q", 2), CReg("c", 2), CReg("d", 2)
    return Main(q, c, d, For(i, 0, 2).Do(d[i].set(c[i])))


def test_symbolic_classical_assignment_preserves_guppy_output() -> None:
    """Pin the complete Guppy output supported before the conversion regression."""
    expected = (
        "from guppylang import guppy\n"
        "from guppylang.std.builtins import array, owned, output\n"
        "from guppylang.std.mem import mem_swap\n"
        "from guppylang.std.quantum import discard, measure, qubit\n"
        "from guppylang.std.quantum.functional import ch, crz, cx, cy, cz, h, reset, rx, ry, rz, "
        "s, sdg, t, tdg, v, vdg, x, y, z\n"
        "from guppylang.std.angles import angle\n"
        "from guppylang.std.qsystem.functional import zz_phase\n"
        "\n"
        "@guppy\n"
        "def main(q: array[qubit, 2] @ owned) -> None:\n"
        "    q_0, q_1 = q\n"
        "    c = array(False, False)\n"
        "    d = array(False, False)\n"
        "    for i in range(0, 2):\n"
        "        d[i] = c[i]\n"
        "    discard(q_0)\n"
        "    discard(q_1)"
    )
    assert SlrConverter(symbolic_classical_assignment_program()).guppy() == expected


@pytest.mark.parametrize("backend", ["qasm", "qir"])
def test_symbolic_classical_assignment_preserves_backend_errors(backend) -> None:
    expected = {
        "qasm": "QASM codegen: For body references loop variable 'i'; binding it is not supported.",
        "qir": (
            "QIR codegen: assignment target 'd[i]' requires a static integer index; "
            "loop-variable indices are not supported."
        ),
    }
    with pytest.raises(NotImplementedError) as error:
        getattr(SlrConverter(symbolic_classical_assignment_program()), backend)()
    assert str(error.value) == expected[backend]


@pytest.mark.parametrize("backend", ["qasm", "qir", "guppy"])
def test_symbolic_measurement_backend_errors(backend) -> None:
    q, c, i = QReg("q", 1), CReg("c", 2), LoopVar("i")
    program = Main(q, c, For(i, 0, 2).Do(qb.Measure(q[0]) > c[i]))
    error = GuppyCodegenError if backend == "guppy" else NotImplementedError
    message = (
        "AST -> Guppy v1 does not support symbolic LoopVar indexing"
        if backend == "guppy"
        else r"For loop variable 'i'.*symbolic bit indices"
    )
    with pytest.raises(error, match=message):
        getattr(SlrConverter(program), backend)()
