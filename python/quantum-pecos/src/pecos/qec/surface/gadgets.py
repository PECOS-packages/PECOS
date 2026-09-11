# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Physical surface-memory gadgets, independent of their rendering."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto

from pecos.qec.surface.circuit_builder import OpType, QubitAllocation, SurfaceCircuitStep
from pecos.qec.surface.patch import SurfacePatch
from pecos.qec.surface.schedule import compute_cnot_schedule


class GadgetKind(Enum):
    """Roles of reusable surface-code functions."""

    PREP = auto()
    INIT_SYNDROME = auto()
    SYNDROME_ROUND = auto()
    MEASURE_OUT = auto()
    LOGICAL_PAULI = auto()


@dataclass(frozen=True)
class Gadget:
    """A physical definition with register allocation and interface dimensions."""

    kind: GadgetKind
    name: str
    steps: tuple[SurfaceCircuitStep, ...]
    allocation: QubitAllocation
    dimensions: tuple[int, int]
    basis: str | None


def default_allocation(patch: SurfacePatch) -> QubitAllocation:
    """Allocate data, then dedicated X and Z ancillas in register order."""
    geom = patch.geometry
    n = geom.num_data
    nx = len(geom.x_stabilizers)
    nz = len(geom.z_stabilizers)
    return QubitAllocation(list(range(n)), list(range(n, n + nx)), list(range(n + nx, n + nx + nz)))


def prep_gadget(patch: SurfacePatch, allocation: QubitAllocation, *, basis: str) -> Gadget:
    """Prepare data in the requested memory basis."""
    name = f"prep_{basis.lower()}_basis"
    steps = [SurfaceCircuitStep(OpType.COMMENT, label=name)]
    steps.extend(SurfaceCircuitStep(OpType.ALLOC, [q], f"data[{i}]") for i, q in enumerate(allocation.data_qubits))
    if basis.upper() == "X":
        steps.extend(SurfaceCircuitStep(OpType.H, [q]) for q in allocation.data_qubits)
    steps.append(SurfaceCircuitStep(OpType.TICK))
    return Gadget(GadgetKind.PREP, name, tuple(steps), allocation, (patch.dx, patch.dz), basis.upper())


def _ancilla_steps(
    patch: SurfacePatch,
    allocation: QubitAllocation,
    family: str,
    op: OpType,
) -> list[SurfaceCircuitStep]:
    stabilizers = patch.geometry.x_stabilizers if family == "X" else patch.geometry.z_stabilizers
    qubits = allocation.x_ancilla_qubits if family == "X" else allocation.z_ancilla_qubits
    prefix = "s" if op == OpType.MEASURE else "a"
    return [SurfaceCircuitStep(op, [qubits[s.index]], f"{prefix}{family.lower()}{s.index}") for s in stabilizers]


def _hadamards(patch: SurfacePatch, allocation: QubitAllocation) -> list[SurfaceCircuitStep]:
    return [
        SurfaceCircuitStep(OpType.COMMENT, label="Hadamard on X ancillas"),
        *_ancilla_steps(patch, allocation, "X", OpType.H),
    ]


def _cx_steps(
    patch: SurfacePatch,
    allocation: QubitAllocation,
    family: str | None = None,
    *,
    round_order: str | Sequence[int] | None = None,
) -> list[SurfaceCircuitStep]:
    steps = []
    for index, layer in enumerate(compute_cnot_schedule(patch, round_order=round_order)):
        steps.append(SurfaceCircuitStep(OpType.COMMENT, label=f"CX round {index + 1}"))
        for kind, stab, data in layer:
            if family is not None and kind != family:
                continue
            operands = (
                [allocation.x_ancilla_qubits[stab], allocation.data_qubits[data]]
                if kind == "X"
                else [allocation.data_qubits[data], allocation.z_ancilla_qubits[stab]]
            )
            steps.append(SurfaceCircuitStep(OpType.CX, operands, f"{kind}{stab}"))
        steps.append(SurfaceCircuitStep(OpType.TICK))
    return steps


def init_syndrome_gadget(
    patch: SurfacePatch,
    allocation: QubitAllocation,
    *,
    basis: str,
    round_order: str | Sequence[int] | None = None,
) -> Gadget:
    """Establish the complementary stabilizer signs after data preparation."""
    family = "X" if basis.upper() == "Z" else "Z"
    steps = [SurfaceCircuitStep(OpType.COMMENT, label=f"init_{family.lower()}_syndrome")]
    steps.extend(_ancilla_steps(patch, allocation, family, OpType.ALLOC))
    if family == "X":
        steps.extend(_hadamards(patch, allocation))
    steps.append(SurfaceCircuitStep(OpType.TICK))
    steps.extend(_cx_steps(patch, allocation, family, round_order=round_order))
    if family == "X":
        steps.extend(_hadamards(patch, allocation))
    steps.append(SurfaceCircuitStep(OpType.COMMENT, label="Measure ancillas"))
    steps.extend(_ancilla_steps(patch, allocation, family, OpType.MEASURE))
    steps.append(SurfaceCircuitStep(OpType.TICK))
    return Gadget(
        GadgetKind.INIT_SYNDROME,
        f"init_{basis.lower()}_basis",
        tuple(steps),
        allocation,
        (patch.dx, patch.dz),
        basis.upper(),
    )


def syndrome_round_gadget(
    patch: SurfacePatch,
    allocation: QubitAllocation,
    *,
    round_index: int,
    round_order: str | Sequence[int] | None = None,
) -> Gadget:
    """Extract one full syndrome in the four-round windmill schedule."""
    steps = [SurfaceCircuitStep(OpType.COMMENT, label=f"syndrome_extraction round {round_index + 1}")]
    steps.extend(_ancilla_steps(patch, allocation, "X", OpType.ALLOC))
    steps.extend(_ancilla_steps(patch, allocation, "Z", OpType.ALLOC))
    steps.extend(_hadamards(patch, allocation))
    steps.append(SurfaceCircuitStep(OpType.TICK))
    steps.extend(_cx_steps(patch, allocation, round_order=round_order))
    steps.extend(_hadamards(patch, allocation))
    steps.append(SurfaceCircuitStep(OpType.COMMENT, label="Measure ancillas"))
    steps.extend(_ancilla_steps(patch, allocation, "X", OpType.MEASURE))
    steps.extend(_ancilla_steps(patch, allocation, "Z", OpType.MEASURE))
    steps.append(SurfaceCircuitStep(OpType.TICK))
    return Gadget(
        GadgetKind.SYNDROME_ROUND,
        "syndrome_extraction",
        tuple(steps),
        allocation,
        (patch.dx, patch.dz),
        None,
    )


def measure_out_gadget(patch: SurfacePatch, allocation: QubitAllocation, *, basis: str) -> Gadget:
    """Destructively measure all data in the memory basis."""
    name = f"measure_{basis.lower()}_basis"
    steps = [SurfaceCircuitStep(OpType.COMMENT, label=name)]
    if basis.upper() == "X":
        steps.extend(SurfaceCircuitStep(OpType.H, [q]) for q in allocation.data_qubits)
    steps.extend(SurfaceCircuitStep(OpType.MEASURE, [q], f"final[{i}]") for i, q in enumerate(allocation.data_qubits))
    return Gadget(GadgetKind.MEASURE_OUT, name, tuple(steps), allocation, (patch.dx, patch.dz), basis.upper())


def logical_pauli_gadget(patch: SurfacePatch, allocation: QubitAllocation, *, pauli: str) -> Gadget:
    """Apply the geometry's logical Pauli string."""
    logical = patch.geometry.logical_x if pauli.upper() == "X" else patch.geometry.logical_z
    steps = (
        tuple(SurfaceCircuitStep(OpType[pauli.upper()], [allocation.data_qubits[q]]) for q in logical.data_qubits)
        if logical
        else ()
    )
    return Gadget(
        GadgetKind.LOGICAL_PAULI,
        f"apply_logical_{pauli.lower()}",
        steps,
        allocation,
        (patch.dx, patch.dz),
        pauli.upper(),
    )


def memory_gadgets(
    patch: SurfacePatch,
    num_rounds: int,
    basis: str,
    *,
    allocation: QubitAllocation | None = None,
    round_order: str | Sequence[int] | None = None,
) -> list[Gadget]:
    """Compose the contiguous physical definitions of a memory experiment."""
    if allocation is None:
        allocation = default_allocation(patch)
    return [
        prep_gadget(patch, allocation, basis=basis),
        init_syndrome_gadget(patch, allocation, basis=basis, round_order=round_order),
        *(syndrome_round_gadget(patch, allocation, round_index=i, round_order=round_order) for i in range(num_rounds)),
        measure_out_gadget(patch, allocation, basis=basis),
    ]
