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

"""Allocator-wide preparation agrees with explicit-slot lowering across backends."""

import re
from dataclasses import replace

import pytest
from pecos.slr.ast.codegen import ast_to_qasm, ast_to_qir, ast_to_stim
from pecos.slr.ast.nodes import AllocatorDecl, GateKind, GateOp, MeasureOp, PrepareOp, Program, SlotRef


@pytest.fixture(params=["PZ", "PNZ", "PX", "PNX", "PY", "PNY"])
def prepared_program(request, child_allocator):
    """Prepare known eigenstates, then rotate them to deterministic Z outcomes."""
    basis = request.param
    declarations = (AllocatorDecl(name="q", capacity=3),)
    name, size = "q", 3
    if child_allocator:
        declarations += (
            AllocatorDecl(name="padding", capacity=1, parent="q", offset=0),
            AllocatorDecl(name="child", capacity=2, parent="q", offset=1),
        )
        name, size = "child", 2
    targets = tuple(SlotRef(allocator=name, index=i) for i in range(size))
    body = [GateOp(gate=GateKind.X, targets=targets), PrepareOp(allocator=name, basis=basis)]
    if basis in {"PY", "PNY"}:
        body.append(GateOp(gate=GateKind.SZdg, targets=targets))
    if basis in {"PX", "PNX", "PY", "PNY"}:
        body.append(GateOp(gate=GateKind.H, targets=targets))
    body.append(MeasureOp(targets=targets))
    program = Program(name="prepare_all", declarations=declarations, body=tuple(body))
    explicit = replace(
        program,
        body=tuple(replace(op, slots=tuple(range(size))) if isinstance(op, PrepareOp) else op for op in body),
    )
    expected_bit = int(basis in {"PNZ", "PNX", "PNY"})
    return program, explicit, expected_bit


@pytest.fixture(params=[False, True], ids=["root", "child"])
def child_allocator(request):
    """Cover both root capacities and child capacities/offsets."""
    return request.param


def test_stim_prepare_all_matches_qasm(prepared_program) -> None:
    """Every QASM reset target is reset in Stim before deterministic measurement."""
    program, explicit, expected_bit = prepared_program
    expected_targets = [int(index) for index in re.findall(r"reset q\[(\d+)\];", ast_to_qasm(explicit))]
    circuit = ast_to_stim(program)
    resets = [
        target.value for instruction in circuit if instruction.name == "R" for target in instruction.targets_copy()
    ]
    assert resets == expected_targets
    assert circuit.num_measurements == len(expected_targets)
    assert (circuit.compile_sampler(seed=42).sample(3) == expected_bit).all()
    assert circuit == ast_to_stim(explicit)


def test_qir_prepare_all_matches_qasm(prepared_program) -> None:
    """Reset calls use the QASM target indices, with the explicit-slot basis tail."""
    program, explicit, _expected_bit = prepared_program
    expected_targets = [int(index) for index in re.findall(r"reset q\[(\d+)\];", ast_to_qasm(explicit))]
    llvm_ir = ast_to_qir(program)
    resets = [line for line in llvm_ir.splitlines() if "call void @__quantum__qis__reset__body" in line]
    assert len(resets) == len(expected_targets)
    for line, index in zip(resets, expected_targets, strict=True):
        pointer = "ptr null" if index == 0 else f"ptr inttoptr (i64 {index} to ptr)"
        assert line.strip() == f"call void @__quantum__qis__reset__body({pointer})"
    assert llvm_ir == ast_to_qir(explicit)


@pytest.mark.parametrize("generate", [ast_to_stim, ast_to_qir], ids=["stim", "qir"])
def test_prepare_all_rejects_unknown_allocator(generate) -> None:
    """An allocator-wide prepare needs a known capacity in every fixed-size backend."""
    program = Program(name="unknown", body=(PrepareOp(allocator="missing"),))
    with pytest.raises(ValueError, match=r"prepare_all.*missing.*declare.*allocator"):
        generate(program)
