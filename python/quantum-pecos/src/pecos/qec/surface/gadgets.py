# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Physical surface-code gadgets, independent of their rendering."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Literal

from pecos.qec.surface.circuit_builder import OpType, QubitAllocation, SurfaceCircuitStep
from pecos.qec.surface.layouts.rotated_lattice import rotated_id_to_position
from pecos.qec.surface.patch import Stabilizer, SurfacePatch
from pecos.qec.surface.schedule import compute_cnot_schedule

_FOLD_AFTER_CX_LAYER = 2


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
    fold: Literal["S", "SDG"] | None = None


def default_allocation(patch: SurfacePatch) -> QubitAllocation:
    """Allocate data, then dedicated X and Z ancillas in register order."""
    geom = patch.geometry
    n = geom.num_data
    nx = len(geom.x_stabilizers)
    nz = len(geom.z_stabilizers)
    return QubitAllocation(list(range(n)), list(range(n, n + nx)), list(range(n + nx, n + nx + nz)))


def _normalize_basis(basis: str, allowed: tuple[str, ...]) -> str:
    basis = basis.upper()
    if basis not in allowed:
        msg = f"Unsupported basis {basis!r}; expected {', '.join(allowed)}"
        raise ValueError(msg)
    return basis


def prep_gadget(patch: SurfacePatch, allocation: QubitAllocation, *, basis: str) -> Gadget:
    """Prepare a product Z, X, or Y eigenstate on all data qubits.

    Y uses H then SZ. Subsequent syndrome projection determines the sign
    of its encoded logical Y eigenstate for odd dx and dz only; neither
    check family is initially deterministic on this product state.
    """
    basis = _normalize_basis(basis, ("X", "Y", "Z"))
    name = f"prep_{basis.lower()}_basis"
    steps = [SurfaceCircuitStep(OpType.COMMENT, label=name)]
    steps.extend(SurfaceCircuitStep(OpType.ALLOC, [q], f"data[{i}]") for i, q in enumerate(allocation.data_qubits))
    if basis.upper() in {"X", "Y"}:
        steps.extend(SurfaceCircuitStep(OpType.H, [q]) for q in allocation.data_qubits)
    if basis.upper() == "Y":
        steps.extend(SurfaceCircuitStep(OpType.SZ, [q]) for q in allocation.data_qubits)
    steps.append(SurfaceCircuitStep(OpType.TICK))
    return Gadget(GadgetKind.PREP, name, tuple(steps), (allocation,), (patch.dx, patch.dz), basis.upper())


def _ancilla_register(
    patch: SurfacePatch,
    allocation: QubitAllocation,
    family: str,
) -> tuple[list[Stabilizer], list[int]]:
    stabilizers = patch.geometry.x_stabilizers if family == "X" else patch.geometry.z_stabilizers
    stabilizers = sorted(stabilizers, key=lambda s: s.index)
    qubits = allocation.x_ancilla_qubits if family == "X" else allocation.z_ancilla_qubits
    if {s.index for s in stabilizers} != set(range(len(qubits))):
        msg = f"{family} stabilizer indices must cover the ancilla register positions"
        raise ValueError(msg)
    return stabilizers, qubits


def _ancilla_steps(
    patch: SurfacePatch,
    allocation: QubitAllocation,
    family: str,
    op: OpType,
) -> list[SurfaceCircuitStep]:
    stabilizers, qubits = _ancilla_register(patch, allocation, family)
    prefix = "s" if op == OpType.MEASURE else "a"
    return [SurfaceCircuitStep(op, [qubits[s.index]], f"{prefix}{family.lower()}{s.index}") for s in stabilizers]


def _hadamards(patch: SurfacePatch, allocation: QubitAllocation, family: str = "X") -> list[SurfaceCircuitStep]:
    return [
        SurfaceCircuitStep(OpType.COMMENT, label=f"Hadamard on {family} ancillas"),
        *_ancilla_steps(patch, allocation, family, OpType.H),
    ]


def _cx_layers(
    patch: SurfacePatch,
    allocation: QubitAllocation,
    family: str | None = None,
    *,
    round_order: str | Sequence[int] | None = None,
    x_z_swapped: bool = False,
) -> list[list[SurfaceCircuitStep]]:
    _, x_ancillas = _ancilla_register(patch, allocation, "X")
    _, z_ancillas = _ancilla_register(patch, allocation, "Z")
    layers = []
    for index, layer in enumerate(compute_cnot_schedule(patch, round_order=round_order)):
        steps = [SurfaceCircuitStep(OpType.COMMENT, label=f"CX round {index + 1}")]
        for kind, stab, data in layer:
            if family is not None and kind != family:
                continue
            operands = (
                [x_ancillas[stab], allocation.data_qubits[data]]
                if kind == "X"
                else [allocation.data_qubits[data], z_ancillas[stab]]
            )
            if x_z_swapped:
                operands.reverse()
            steps.append(SurfaceCircuitStep(OpType.CX, operands, f"{kind}{stab}"))
        steps.append(SurfaceCircuitStep(OpType.TICK))
        layers.append(steps)
    return layers


def init_syndrome_gadget(
    patch: SurfacePatch,
    allocation: QubitAllocation,
    *,
    basis: str,
    round_order: str | Sequence[int] | None = None,
    x_z_swapped: bool = False,
) -> Gadget:
    """Establish the complementary stabilizer signs after data preparation."""
    basis = _normalize_basis(basis, ("X", "Z"))
    family = "X" if basis.upper() == "Z" else "Z"
    if x_z_swapped and patch.dx != patch.dz:
        msg = "init_syndrome_gadget requires a square patch when x_z_swapped (dx == dz)"
        raise ValueError(msg)
    h_family = "Z" if x_z_swapped else "X"
    if x_z_swapped:
        family = "Z" if family == "X" else "X"
    steps = [SurfaceCircuitStep(OpType.COMMENT, label=f"init_{family.lower()}_syndrome")]
    steps.extend(_ancilla_steps(patch, allocation, family, OpType.ALLOC))
    if family == h_family:
        steps.extend(_hadamards(patch, allocation, h_family))
    steps.append(SurfaceCircuitStep(OpType.TICK))
    steps.extend(
        step
        for layer in _cx_layers(patch, allocation, family, round_order=round_order, x_z_swapped=x_z_swapped)
        for step in layer
    )
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
    layers = _cx_layers(patch, allocation, round_order=round_order, x_z_swapped=x_z_swapped)
    return _syndrome_round(patch, allocation, layers, round_index=round_index, x_z_swapped=x_z_swapped)


def _syndrome_round(
    patch: SurfacePatch,
    allocation: QubitAllocation,
    layers: list[list[SurfaceCircuitStep]],
    *,
    round_index: int,
    x_z_swapped: bool,
    name: str = "syndrome_extraction",
    fold: Literal["S", "SDG"] | None = None,
) -> Gadget:
    if x_z_swapped and patch.dx != patch.dz:
        # Transversal H needs a square patch, so a swapped rectangle has no producer and no
        # Guppy syndrome struct of its own.
        msg = "syndrome_round_gadget requires a square patch when x_z_swapped (dx == dz)"
        raise ValueError(msg)
    steps = [SurfaceCircuitStep(OpType.COMMENT, label=f"syndrome_extraction round {round_index + 1}")]
    families = ("Z", "X") if x_z_swapped else ("X", "Z")
    for family in families:
        steps.extend(_ancilla_steps(patch, allocation, family, OpType.ALLOC))
    steps.extend(_hadamards(patch, allocation, families[0]))
    steps.append(SurfaceCircuitStep(OpType.TICK))
    steps.extend(step for layer in layers for step in layer)
    steps.extend(_hadamards(patch, allocation, families[0]))
    steps.append(SurfaceCircuitStep(OpType.COMMENT, label="Measure ancillas"))
    for family in families:
        steps.extend(_ancilla_steps(patch, allocation, family, OpType.MEASURE))
    steps.append(SurfaceCircuitStep(OpType.TICK))
    return Gadget(
        GadgetKind.SYNDROME_ROUND,
        name + ("_swapped" if x_z_swapped else ""),
        tuple(steps),
        (allocation,),
        (patch.dx, patch.dz),
        None,
        x_z_swapped=x_z_swapped,
        fold=fold,
    )


def fold_s_round_gadget(
    patch: SurfacePatch,
    allocation: QubitAllocation,
    *,
    round_index: int,
    x_z_swapped: bool = False,
    dagger: bool = False,
) -> Gadget:
    """Apply logical S inside a default syndrome round on a square rotated patch.

    After CX layer 2, transpose (x, y) -> (y, x) exchanges the entangled
    block's code X/Z subgroups. Apply CZ to every exchanged pair of data
    or bulk ancillas. Diagonal data sites have odd coordinates and bulk
    ancilla sites even, so S on data and S-dagger on ancillas alternate
    along the diagonal. Exterior weight-2 check ancillas are disentangled
    and untouched. Distance must be at least 2 to have syndrome checks.
    The same coordinate rule applies in the current X/Z orientation.

    The exact round flow is X_L -> +Y_L * product(current Z checks) and
    Z_L -> Z_L, with +Y_L = i X_L Z_L, SparseStab's Y = iXZ convention,
    and PECOS SZ = diag(1, i). For an X-prepared patch the output logical
    Y sign is (-1)**parity(round Z outcomes). With dagger=True all fixed
    point phases reverse, giving -Y_L and the opposite frame sign.

    X records are not bare X checks: on input, bottom-row bulk X ancillas
    measure their X check times the left-boundary Z check at (0, x_j); other
    X records measure bare checks. On output, an X record together with
    the Z record at (y_j + 2, x_j) certifies X check j. If that partner is
    absent the X record alone certifies the check. Coordinates here use
    the current frame (transpose them when x_z_swapped=True).

    Under circuit noise the X-sector fault distance is reduced: a Y fault
    before the fold produces a Z pair on a mirror pair. Chen, Chen, Lu, Pan,
    arXiv:2412.01391 (https://arxiv.org/abs/2412.01391), observe two to three
    times the memory's logical error rate for their separated S-round benchmark.
    See also the half-cycle construction of McEwen, Bacon, Gidney, arXiv:2302.02192
    (https://arxiv.org/abs/2302.02192). Builder and detector integration
    are separate from this physical gadget.

    Raises:
        ValueError: For non-rotated, rectangular, distance-1, or malformed patches,
            or a CX schedule that does not have four layers.
    """
    if not patch.rotated:
        msg = "fold_s_round_gadget requires a rotated patch"
        raise ValueError(msg)
    if patch.dx != patch.dz:
        msg = "fold_s_round_gadget requires a square patch (dx=dz)"
        raise ValueError(msg)
    if patch.dx < 2:
        msg = "fold_s_round_gadget requires distance at least 2 to have syndrome checks"
        raise ValueError(msg)

    positions = {i: rotated_id_to_position(i, patch.dx) for i in range(patch.geometry.num_data)}
    by_position = {position: allocation.data_qubits[i] for i, position in positions.items()}
    # Supports identify bulk centres without relying on placeholder stabilizer positions.
    for family in ("X", "Z"):
        stabilizers, ancillas = _ancilla_register(patch, allocation, family)
        for stabilizer in stabilizers:
            if len(stabilizer.data_qubits) == 4:
                x_sum = sum(positions[q][0] for q in stabilizer.data_qubits)
                y_sum = sum(positions[q][1] for q in stabilizer.data_qubits)
                if x_sum % 4 or y_sum % 4:
                    msg = "fold_s_round_gadget requires bulk-centre coordinate sums divisible by 4"
                    raise ValueError(msg)
                by_position[x_sum // 4, y_sum // 4] = ancillas[stabilizer.index]

    if len(by_position) != patch.dx**2 + (patch.dx - 1) ** 2:
        msg = "fold_s_round_gadget requires d*d + (d-1)**2 distinct data and bulk-ancilla sites"
        raise ValueError(msg)

    label = "fold-transversal S-dagger layer" if dagger else "fold-transversal S layer"
    fold = [SurfaceCircuitStep(OpType.COMMENT, label=label)]
    for (x, y), qubit in sorted(by_position.items()):
        if (y, x) not in by_position:
            msg = f"fold_s_round_gadget missing transpose partner for site {(x, y)}"
            raise ValueError(msg)
        if x < y:
            fold.append(SurfaceCircuitStep(OpType.CZ, [qubit, by_position[y, x]]))
        elif x == y:
            op = OpType.SZ if (x % 2 == 1) != dagger else OpType.SZDG
            fold.append(SurfaceCircuitStep(op, [qubit]))
    fold.append(SurfaceCircuitStep(OpType.TICK))
    layers = _cx_layers(patch, allocation, x_z_swapped=x_z_swapped)
    if len(layers) != 4:
        msg = "fold_s_round_gadget requires four CX layers in the default schedule"
        raise ValueError(msg)
    # Between CX layers 2 and 3 the half-cycle state is the unrotated code;
    # round_order is deliberately absent because the fold depends on the default order.
    layers.insert(_FOLD_AFTER_CX_LAYER, fold)
    return _syndrome_round(
        patch,
        allocation,
        layers,
        round_index=round_index,
        x_z_swapped=x_z_swapped,
        name="syndrome_extraction_fold_sdg" if dagger else "syndrome_extraction_fold_s",
        fold="SDG" if dagger else "S",
    )


def measure_out_gadget(patch: SurfacePatch, allocation: QubitAllocation, *, basis: str) -> Gadget:
    """Destructively measure all data in the memory basis."""
    if basis.upper() == "Y":
        msg = "Y readout is unsupported"
        raise NotImplementedError(msg)
    basis = _normalize_basis(basis, ("X", "Z"))
    name = f"measure_{basis.lower()}_basis"
    steps = [SurfaceCircuitStep(OpType.COMMENT, label=name)]
    if basis.upper() == "X":
        steps.extend(SurfaceCircuitStep(OpType.H, [q]) for q in allocation.data_qubits)
    steps.extend(SurfaceCircuitStep(OpType.MEASURE, [q], f"final[{i}]") for i, q in enumerate(allocation.data_qubits))
    return Gadget(GadgetKind.MEASURE_OUT, name, tuple(steps), (allocation,), (patch.dx, patch.dz), basis.upper())


def logical_pauli_gadget(patch: SurfacePatch, allocation: QubitAllocation, *, pauli: str) -> Gadget:
    """Apply the geometry's logical Pauli string."""
    pauli = _normalize_basis(pauli, ("X", "Z"))
    logical = patch.geometry.logical_x if pauli.upper() == "X" else patch.geometry.logical_z
    if logical is None:
        msg = f"Patch has no logical {pauli} operator"
        raise ValueError(msg)
    steps = tuple(SurfaceCircuitStep(OpType[pauli], [allocation.data_qubits[q]]) for q in logical.data_qubits)
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
