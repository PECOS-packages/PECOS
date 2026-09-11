# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Physical surface-code gadgets, independent of their rendering."""

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
    TRANSVERSAL = auto()
    TWO_PATCH = auto()


@dataclass(frozen=True)
class Gadget:
    """A physical definition with register allocation and interface dimensions."""

    kind: GadgetKind
    name: str
    steps: tuple[SurfaceCircuitStep, ...]
    allocations: tuple[QubitAllocation, ...]
    dimensions: tuple[int, int]
    basis: str | None
    x_z_swapped: bool = False


def default_allocation(patch: SurfacePatch) -> QubitAllocation:
    """Allocate data, then dedicated X and Z ancillas in register order."""
    geom = patch.geometry
    n = geom.num_data
    nx = len(geom.x_stabilizers)
    nz = len(geom.z_stabilizers)
    return QubitAllocation(list(range(n)), list(range(n, n + nx)), list(range(n + nx, n + nx + nz)))


def prep_gadget(patch: SurfacePatch, allocation: QubitAllocation, *, basis: str) -> Gadget:
    """Prepare a product Z, X, or Y eigenstate on all data qubits.

    Y uses H then SZ. Subsequent syndrome projection determines the sign
    of its encoded logical Y eigenstate; neither check family is initially
    deterministic on this product state.
    """
    name = f"prep_{basis.lower()}_basis"
    steps = [SurfaceCircuitStep(OpType.COMMENT, label=name)]
    steps.extend(SurfaceCircuitStep(OpType.ALLOC, [q], f"data[{i}]") for i, q in enumerate(allocation.data_qubits))
    if basis.upper() in {"X", "Y"}:
        steps.extend(SurfaceCircuitStep(OpType.H, [q]) for q in allocation.data_qubits)
    if basis.upper() == "Y":
        steps.extend(SurfaceCircuitStep(OpType.SZ, [q]) for q in allocation.data_qubits)
    steps.append(SurfaceCircuitStep(OpType.TICK))
    return Gadget(GadgetKind.PREP, name, tuple(steps), (allocation,), (patch.dx, patch.dz), basis.upper())


def _ancilla_steps(
    patch: SurfacePatch,
    allocation: QubitAllocation,
    family: str,
    op: OpType,
) -> list[SurfaceCircuitStep]:
    stabilizers = patch.geometry.x_stabilizers if family == "X" else patch.geometry.z_stabilizers
    stabilizers = sorted(stabilizers, key=lambda s: s.index)
    qubits = allocation.x_ancilla_qubits if family == "X" else allocation.z_ancilla_qubits
    prefix = "s" if op == OpType.MEASURE else "a"
    return [SurfaceCircuitStep(op, [qubits[s.index]], f"{prefix}{family.lower()}{s.index}") for s in stabilizers]


def _hadamards(patch: SurfacePatch, allocation: QubitAllocation, family: str = "X") -> list[SurfaceCircuitStep]:
    return [
        SurfaceCircuitStep(OpType.COMMENT, label=f"Hadamard on {family} ancillas"),
        *_ancilla_steps(patch, allocation, family, OpType.H),
    ]


def _cx_steps(
    patch: SurfacePatch,
    allocation: QubitAllocation,
    family: str | None = None,
    *,
    round_order: str | Sequence[int] | None = None,
    x_z_swapped: bool = False,
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
            if x_z_swapped:
                operands.reverse()
            steps.append(SurfaceCircuitStep(OpType.CX, operands, f"{kind}{stab}"))
        steps.append(SurfaceCircuitStep(OpType.TICK))
    return steps


def init_syndrome_gadget(
    patch: SurfacePatch,
    allocation: QubitAllocation,
    *,
    basis: str,
    round_order: str | Sequence[int] | None = None,
    x_z_swapped: bool = False,
) -> Gadget:
    """Establish the complementary stabilizer signs after data preparation."""
    family = "X" if basis.upper() == "Z" else "Z"
    h_family = "Z" if x_z_swapped else "X"
    if x_z_swapped:
        family = "Z" if family == "X" else "X"
    steps = [SurfaceCircuitStep(OpType.COMMENT, label=f"init_{family.lower()}_syndrome")]
    steps.extend(_ancilla_steps(patch, allocation, family, OpType.ALLOC))
    if family == h_family:
        steps.extend(_hadamards(patch, allocation, h_family))
    steps.append(SurfaceCircuitStep(OpType.TICK))
    steps.extend(_cx_steps(patch, allocation, family, round_order=round_order, x_z_swapped=x_z_swapped))
    if family == h_family:
        steps.extend(_hadamards(patch, allocation, h_family))
    steps.append(SurfaceCircuitStep(OpType.COMMENT, label="Measure ancillas"))
    steps.extend(_ancilla_steps(patch, allocation, family, OpType.MEASURE))
    steps.append(SurfaceCircuitStep(OpType.TICK))
    return Gadget(
        GadgetKind.INIT_SYNDROME,
        f"init_{basis.lower()}_basis" + ("_swapped" if x_z_swapped else ""),
        tuple(steps),
        (allocation,),
        (patch.dx, patch.dz),
        basis.upper(),
        x_z_swapped=x_z_swapped,
    )


def syndrome_round_gadget(
    patch: SurfacePatch,
    allocation: QubitAllocation,
    *,
    round_index: int,
    round_order: str | Sequence[int] | None = None,
    x_z_swapped: bool = False,
) -> Gadget:
    """Extract a full windmill syndrome, reversing both CX families after H.

    In the swapped orientation base Z ancillas carry current X checks,
    receive the Hadamards, and are allocated and measured first. Labels
    always refer to physical register slots by stabilizer index.
    """
    steps = [SurfaceCircuitStep(OpType.COMMENT, label=f"syndrome_extraction round {round_index + 1}")]
    families = ("Z", "X") if x_z_swapped else ("X", "Z")
    for family in families:
        steps.extend(_ancilla_steps(patch, allocation, family, OpType.ALLOC))
    steps.extend(_hadamards(patch, allocation, families[0]))
    steps.append(SurfaceCircuitStep(OpType.TICK))
    steps.extend(_cx_steps(patch, allocation, round_order=round_order, x_z_swapped=x_z_swapped))
    steps.extend(_hadamards(patch, allocation, families[0]))
    steps.append(SurfaceCircuitStep(OpType.COMMENT, label="Measure ancillas"))
    for family in families:
        steps.extend(_ancilla_steps(patch, allocation, family, OpType.MEASURE))
    steps.append(SurfaceCircuitStep(OpType.TICK))
    return Gadget(
        GadgetKind.SYNDROME_ROUND,
        "syndrome_extraction" + ("_swapped" if x_z_swapped else ""),
        tuple(steps),
        (allocation,),
        (patch.dx, patch.dz),
        None,
        x_z_swapped=x_z_swapped,
    )


def measure_out_gadget(patch: SurfacePatch, allocation: QubitAllocation, *, basis: str) -> Gadget:
    """Destructively measure all data in the memory basis."""
    if basis.upper() == "Y":
        msg = "Y readout is unsupported"
        raise NotImplementedError(msg)
    name = f"measure_{basis.lower()}_basis"
    steps = [SurfaceCircuitStep(OpType.COMMENT, label=name)]
    if basis.upper() == "X":
        steps.extend(SurfaceCircuitStep(OpType.H, [q]) for q in allocation.data_qubits)
    steps.extend(SurfaceCircuitStep(OpType.MEASURE, [q], f"final[{i}]") for i, q in enumerate(allocation.data_qubits))
    return Gadget(GadgetKind.MEASURE_OUT, name, tuple(steps), (allocation,), (patch.dx, patch.dz), basis.upper())


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
        (allocation,),
        (patch.dx, patch.dz),
        pauli.upper(),
    )


def same_static_geometry(first: SurfacePatch, second: SurfacePatch) -> bool:
    """Whether corresponding data indices carry identical CSS and logical supports."""
    a, b = first.geometry, second.geometry
    return (
        (a.dx, a.dz, a.rotated, a.orientation) == (b.dx, b.dz, b.rotated, b.orientation)
        and {s.index: s.data_qubits for s in a.x_stabilizers} == {s.index: s.data_qubits for s in b.x_stabilizers}
        and {s.index: s.data_qubits for s in a.z_stabilizers} == {s.index: s.data_qubits for s in b.z_stabilizers}
        and a.logical_x == b.logical_x
        and a.logical_z == b.logical_z
    )


def transversal_layer_gadget(patch: SurfacePatch, allocation: QubitAllocation, *, gate: str) -> Gadget:
    """Apply H, SZ, or SZDG to every data qubit.

    H exchanges X and Z checks, the textbook transversal CSS construction.
    The physical SZ layers are not logical S gates on this surface code.
    """
    names = {"H": "transversal_h", "SZ": "physical_sz_layer", "SZDG": "physical_szdg_layer"}
    if gate not in names:
        msg = f"Unsupported transversal layer: {gate}"
        raise ValueError(msg)
    if gate == "H" and patch.dx != patch.dz:
        msg = "Transversal H requires a square patch (dx=dz)"
        raise ValueError(msg)
    return Gadget(
        GadgetKind.TRANSVERSAL,
        names[gate],
        tuple(SurfaceCircuitStep(OpType[gate], [q]) for q in allocation.data_qubits),
        (allocation,),
        (patch.dx, patch.dz),
        gate,
    )


def transversal_cx_gadget(
    ctrl_patch: SurfacePatch,
    ctrl_allocation: QubitAllocation,
    tgt_patch: SurfacePatch,
    tgt_allocation: QubitAllocation,
) -> Gadget:
    """Apply control-to-target CX at each data index, the textbook CSS transversal CX."""
    if not same_static_geometry(ctrl_patch, tgt_patch):
        msg = "Transversal CX requires the same static geometry"
        raise ValueError(msg)
    return Gadget(
        GadgetKind.TWO_PATCH,
        "transversal_cx",
        tuple(
            SurfaceCircuitStep(OpType.CX, [ctrl, tgt])
            for ctrl, tgt in zip(ctrl_allocation.data_qubits, tgt_allocation.data_qubits, strict=True)
        ),
        (ctrl_allocation, tgt_allocation),
        (ctrl_patch.dx, ctrl_patch.dz),
        "CX",
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
