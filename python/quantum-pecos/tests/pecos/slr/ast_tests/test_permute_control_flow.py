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

"""A compile-time permutation must not escape runtime control flow (#80, #992)."""

import pytest
import stim
from pecos.slr import Block, CReg, For, If, Main, Permute, QReg, Repeat, While
from pecos.slr.ast import slr_to_ast
from pecos.slr.ast.codegen import generate
from pecos.slr.ast.codegen.guppy_linearity import LinearityError
from pecos.slr.gen_codes.gen_qasm import QASMGenerator
from pecos.slr.gen_codes.gen_qir import QIRGenerator
from pecos.slr.gen_codes.gen_quantum_circuit import QuantumCircuitGenerator
from pecos.slr.gen_codes.gen_stim import StimGenerator
from pecos.slr.qeclib import qubit as qb

BACKENDS = ("qasm", "qir", "stim", "quantum_circuit")
LEGACY = {
    "legacy_qasm": QASMGenerator,
    "legacy_qir": QIRGenerator,
    "legacy_stim": StimGenerator,
    "legacy_quantum_circuit": QuantumCircuitGenerator,
}
# The deprecated StimGenerator rejects If, While and For outright.
LEGACY_WITH_REGIONS = ("legacy_qasm", "legacy_qir", "legacy_quantum_circuit")


def compile_program(program, backend):
    """Exercise the public AST entry point or the deprecated generator directly."""
    if backend in LEGACY:
        generator = LEGACY[backend](_internal=True)
        generator.generate_block(program)
        return generator.get_output()
    return generate(slr_to_ast(program), backend)


@pytest.mark.parametrize("backend", [*BACKENDS, *LEGACY_WITH_REGIONS])
@pytest.mark.parametrize("branch", ["then", "else"])
@pytest.mark.parametrize("nested", [False, True])
def test_conditional_qubit_permute_rejected(backend, branch, nested) -> None:
    """Check both bodies, including a Permute hidden in an unrolled nested block."""
    q, c = QReg("q", 3), CReg("c", 1)
    operation = Permute([q[0], q[1]], [q[1], q[0]])
    if nested:
        operation = Repeat(2).block(Block(operation))
    conditional = If(c == 1)
    if branch == "then":
        conditional.Then(operation)
    else:
        conditional.Then(qb.X(q[0])).Else(operation)
    program = Main(q, c, qb.Measure(q[2]) > c[0], conditional, qb.X(q[0]))
    with pytest.raises(NotImplementedError, match=r"Permute.*If"):
        compile_program(program, backend)


@pytest.mark.parametrize("backend", ["qasm", "qir", "stim", "quantum_circuit", "legacy_qasm"])
@pytest.mark.parametrize("branch", ["then", "else"])
def test_conditional_classical_permute_rejected(backend, branch) -> None:
    """Static classical relabels and unguarded QASM swaps are unsafe too."""
    q, c = QReg("q", 1), CReg("c", 2)
    operation = Permute([c[0], c[1]], [c[1], c[0]])
    conditional = If(c == 1)
    if branch == "then":
        conditional.Then(operation)
    else:
        conditional.Then(qb.X(q[0])).Else(operation)
    with pytest.raises(NotImplementedError, match=r"Permute.*If"):
        compile_program(Main(q, c, qb.Measure(q[0]) > c[0], conditional), backend)


@pytest.mark.parametrize("backend", [*BACKENDS, *LEGACY_WITH_REGIONS])
def test_while_permute_rejected(backend) -> None:
    """Even generators which skip unsupported bodies must reject their Permutes."""
    q, c = QReg("q", 2), CReg("c", 1)
    program = Main(q, c, While(c == 1).Do(Permute([q[0], q[1]], [q[1], q[0]])))
    reason = r"While.*SlrConverter.*qir" if backend == "legacy_qir" else r"Permute.*While"
    with pytest.raises(NotImplementedError, match=reason):
        compile_program(program, backend)


@pytest.mark.parametrize("count", [0, 1, 2, 3])
@pytest.mark.parametrize("nested", [False, True])
def test_stim_repeat_permute_records(count, nested) -> None:
    """#992: every iteration and the following measurement use the current map."""
    q, c = QReg("q", 3), CReg("c", 1)
    body = Block(qb.Measure(q[0]) > c[0], Permute([q[0], q[1]], [q[1], q[0]]))
    if nested:
        body = Repeat(1).block(body)
    program = Main(q, c, qb.X(q[0]), Repeat(count).block(body), qb.Measure(q[2]), qb.Measure(q[0]))
    circuit = stim.Circuit(compile_program(program, "stim"))
    expected = [1 - i % 2 for i in range(count)] + [0, 1 - count % 2]
    assert circuit.compile_sampler().sample(shots=4).tolist() == [expected] * 4


@pytest.mark.parametrize("region", ["Repeat", "For", "If then", "If else", "If nested Repeat"])
def test_guppy_region_permute_must_restore_slot_bindings(region) -> None:
    """Guppy's slot relabel is static, so a region must leave every slot bound as it found it."""
    q, c = QReg("q", 3), CReg("c", 1)
    swap = Permute([q[0], q[1]], [q[1], q[0]])
    body = {
        "Repeat": lambda: Repeat(2).block(swap),
        "For": lambda: For("i", 0, 2).Do(swap),
        "If then": lambda: If(c == 1).Then(swap),
        "If else": lambda: If(c == 1).Then(qb.X(q[0])).Else(swap),
        "If nested Repeat": lambda: If(c == 1).Then(qb.X(q[0])).Else(Repeat(2).block(swap)),
    }[region]()
    program = Main(q, c, qb.Measure(q[2]) > c[0], body, qb.X(q[0]))
    with pytest.raises(LinearityError, match=r"divergent Guppy slot states|changes Guppy slot state"):
        compile_program(program, "guppy")


@pytest.mark.parametrize("branch", ["then", "else", "both"])
def test_legacy_qir_classical_permute_has_valid_runtime_branches(branch) -> None:
    """Classical QIR swaps must never share a branch-local temporary pointer."""
    from llvmlite import binding

    c, flag = CReg("c", 2), CReg("flag", 1)
    operation = Permute([c[0], c[1]], [c[1], c[0]])
    conditional = If(flag == 1).Then(operation if branch != "else" else c[0].set(1))
    if branch != "then":
        conditional.Else(operation)
    code = compile_program(Main(c, flag, conditional), "legacy_qir")
    module = binding.parse_assembly(code)
    module.verify()
    assert "@get_creg_bit" in code
    assert "@set_creg_bit" in code


def test_legacy_stim_repeat_permute_rejected() -> None:
    """The legacy placeholder cannot faithfully lower a permutation in a REPEAT block."""
    q = QReg("q", 2)
    operation = Block(Permute([q[0], q[1]], [q[1], q[0]]))
    with pytest.raises(NotImplementedError, match=r"Permute.*Repeat"):
        compile_program(Main(q, Repeat(2).block(operation)), "legacy_stim")


@pytest.mark.parametrize("loop", ["While", "For"])
def test_legacy_stim_rejects_runtime_and_variable_loops(loop) -> None:
    """Emitting the body once would drop iterations; While used to recurse forever."""
    q, c = QReg("q", 1), CReg("c", 1)
    body = While(c == 1).Do(qb.X(q[0])) if loop == "While" else For("i", 0, 2).Do(qb.X(q[0]))
    with pytest.raises(NotImplementedError, match=f"StimGenerator does not support {loop}"):
        compile_program(Main(q, c, body), "legacy_stim")


def test_issue_992_exact_reproducer() -> None:
    """The reported program measures qubits zero, one, then two."""
    program = Main(
        q := QReg("q", 3),
        c := CReg("c", 1),
        qb.X(q[0]),
        Repeat(2).block(qb.Measure(q[0]) > c[0], Permute([q[0], q[1]], [q[1], q[0]])),
        qb.Measure(q[2]),
    )
    circuit = stim.Circuit(compile_program(program, "stim"))
    assert circuit.compile_sampler().sample(shots=4).tolist() == [[1, 0, 0]] * 4


def test_stim_unrolled_repeat_preserves_measurement_tracking() -> None:
    """Post-loop feed-forward must use the last iteration's measurement record."""
    q, c = QReg("q", 3), CReg("c", 1)
    program = Main(
        q,
        c,
        qb.X(q[0]),
        Repeat(2).block(qb.Measure(q[0]) > c[0], Permute([q[0], q[1]], [q[1], q[0]])),
        If(c[0]).Then(qb.X(q[2])),
        qb.Measure(q[2]),
    )
    circuit = stim.Circuit(compile_program(program, "stim"))
    assert circuit.compile_sampler().sample(shots=4).tolist() == [[1, 0, 0]] * 4


@pytest.mark.parametrize("whole_register", [False, True])
def test_stim_repeat_classical_permute_invalidates_measurement(whole_register) -> None:
    """Stim rejects classical permutations before later record-dependent uses."""
    q, c, d = QReg("q", 1), CReg("c", 1), CReg("d", 1)
    operation = Permute(c, d) if whole_register else Permute([c[0], d[0]], [d[0], c[0]])
    program = Main(q, c, d, qb.Measure(q[0]) > c[0], Repeat(2).block(operation), If(c[0]).Then(qb.X(q[0])))
    with pytest.raises(NotImplementedError, match=r"Permute.*not a known qubit allocator"):
        compile_program(program, "stim")


@pytest.mark.parametrize("backend", ["qasm", "qir", "stim", "quantum_circuit"])
def test_conditional_block_call_permute_rejected(backend) -> None:
    """Flattened reusable blocks inherit the control-flow context at each call."""
    from dataclasses import replace

    from pecos.slr.ast.nodes import (
        AllocatorArg,
        ArrayTypeExpr,
        BlockCall,
        BlockDecl,
        BlockInput,
        IfStmt,
        LiteralExpr,
        PermuteOp,
        QubitTypeExpr,
        ResourceEffect,
    )

    program = slr_to_ast(Main(QReg("q", 2)))
    declaration = BlockDecl(
        name="swap_slots",
        inputs=(
            BlockInput(
                name="input_q",
                effect=ResourceEffect.LIVE_PRESERVED,
                type_expr=ArrayTypeExpr(element=QubitTypeExpr(), size=2),
            ),
        ),
        body=(PermuteOp(sources=("input_q[0]", "input_q[1]"), targets=("input_q[1]", "input_q[0]")),),
    )
    call = BlockCall(callee="swap_slots", arg_bindings=(AllocatorArg(name="q"),))
    program = replace(
        program,
        block_decls=(declaration,),
        body=(IfStmt(condition=LiteralExpr(value=True), then_body=(), else_body=(call,)),),
    )
    with pytest.raises(NotImplementedError, match=r"Permute.*If"):
        generate(program, backend)
