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

"""Quantum and classical declarations must have distinct visible names."""

from functools import partial
from typing import ClassVar

import pytest
from pecos.slr import Block, CReg, For, If, Main, Permute, QAlloc, QReg, SlrConverter
from pecos.slr.ast import slr_to_ast
from pecos.slr.qeclib import qubit as qb
from pecos.slr.qeclib.steane.steane_class import Steane


@pytest.mark.parametrize("kind", ["QReg", "QAlloc", "child"])
@pytest.mark.parametrize("classical_first", [False, True])
@pytest.mark.parametrize("scope", ["main", "nested", "inherited", "siblings", "else", "loop"])
def test_quantum_classical_name_collision(kind, classical_first, scope) -> None:
    parent = QAlloc(4, name="parent")
    quantum = {
        "QReg": lambda: QReg("shared", 2),
        "QAlloc": lambda: QAlloc(2, name="shared"),
        "child": lambda: parent.child(2, name="shared"),
    }[kind]()
    classical = CReg("shared", 2)
    pair = (classical, quantum) if classical_first else (quantum, classical)
    extra = (parent,) if kind == "child" else ()
    if scope == "main":
        program = Main(*extra, *pair)
    elif scope == "nested":
        program = Main(*extra, Block(*pair))
    elif scope == "inherited":
        program = Main(*extra, pair[0], Block(pair[1]))
    elif scope == "siblings":
        program = Main(*extra, Block(pair[0]), Block(pair[1]))
    elif scope == "else":
        condition = CReg("condition", 1)
        program = Main(*extra, condition, If(condition == 0).Then().Else(Block(*pair)))
    else:
        program = Main(*extra, For("i", 0, 2).Do(*pair))
    with pytest.raises(ValueError, match=r"quantum.*classical.*'shared'.*[Rr]ename"):
        slr_to_ast(program)


def test_block_input_register_rejects_quantum_classical_collision() -> None:
    quantum = QReg("shared", 2)
    classical = CReg("shared", 2)

    class UsesQuantum(Block):
        block_inputs: ClassVar[dict[str, str]] = {"input": "live_preserved"}

        def __init__(self) -> None:
            super().__init__()
            self.input = quantum
            self.extend(qb.H(quantum[0]))

    with pytest.raises(ValueError, match=r"quantum.*classical.*'shared'.*[Rr]ename"):
        slr_to_ast(Main(classical, UsesQuantum()))


@pytest.mark.parametrize("backend", ["qasm", "stim", "qir", "quantum_circuit", "guppy"])
@pytest.mark.parametrize("gate", ["h", "cx"])
def test_block_input_formal_name_may_match_a_classical_register(gate, backend) -> None:
    # Steane's H block names its input `q`, and CX names its inputs `q1` and `q2`.
    a, b = Steane("a"), Steane("b")
    program = Main(a, CReg("q", 1), a.h()) if gate == "h" else Main(a, b, CReg("q1", 2), a.cx(b))
    assert getattr(SlrConverter(program), backend)()


@pytest.mark.parametrize("backend", ["ast", "qasm", "stim", "qir", "quantum_circuit", "guppy"])
def test_classical_swap_cannot_relabel_same_named_qubits(backend) -> None:
    qq, cq = QReg("q", 2), CReg("q", 2)
    program = Main(qq, cq, Permute([cq[0], cq[1]], [cq[1], cq[0]]), qb.H(qq[0]))
    convert = partial(slr_to_ast, program) if backend == "ast" else getattr(SlrConverter(program), backend)
    with pytest.raises(ValueError, match=r"quantum.*classical.*'q'.*[Rr]ename"):
        convert()
