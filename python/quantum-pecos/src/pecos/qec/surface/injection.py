# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Raw state injection by logical-string encoding and stabilizer projection.

This is the GHZ-seed construction of Horsman et al., arXiv:1111.4022,
section 4.2, generalized to the patch's logical-X support. It is an encoder,
not a distillation protocol or a fault-tolerant preparation. Its single-qubit
seed error is not suppressed by increasing the patch distance.
"""

from dataclasses import dataclass
from itertools import pairwise

from pecos.qec.surface.circuit_builder import OpType, QubitAllocation, SurfaceCircuitStep
from pecos.qec.surface.gadgets import Gadget, GadgetKind, default_allocation, syndrome_round_gadget
from pecos.qec.surface.patch import SurfacePatch


def injection_correction_supports(patch: SurfacePatch, *, basis: str = "X") -> tuple[tuple[int, ...], ...]:
    """Return opposite-Pauli corrections for checks in ``basis``.

    For X checks solve H_X c_i = e_i and X_L c_i = 0 over GF(2), giving
    Z corrections. For Z checks use H_Z and Z_L, giving X corrections.
    Thus random projection signs are removed without changing logical amplitudes.
    These are encoding byproducts, NOT a decoder for noisy syndrome records.
    Indices in the returned supports refer to the patch's data register.
    """
    if basis not in {"X", "Z"}:
        msg = "Correction check basis must be X or Z"
        raise ValueError(msg)
    checks = patch.geometry.x_stabilizers if basis == "X" else patch.geometry.z_stabilizers
    logical = patch.geometry.logical_x if basis == "X" else patch.geometry.logical_z
    rows = [sum(1 << q for q in check.data_qubits) for check in checks]
    rows.append(sum(1 << q for q in logical.data_qubits))
    rhs = [1 << i for i in range(len(checks))] + [0]
    pivots = []
    for q in range(patch.geometry.num_data):
        pivot = next((r for r in range(len(pivots), len(rows)) if rows[r] & (1 << q)), None)
        if pivot is None:
            continue
        r = len(pivots)
        rows[r], rows[pivot] = rows[pivot], rows[r]
        rhs[r], rhs[pivot] = rhs[pivot], rhs[r]
        for other in range(len(rows)):
            if other != r and rows[other] & (1 << q):
                rows[other] ^= rows[r]
                rhs[other] ^= rhs[r]
        pivots.append(q)
    if len(pivots) != len(rows):
        msg = f"State injection requires independent {basis} checks and logical {basis}"
        raise ValueError(msg)
    return tuple(tuple(q for q, mask in zip(pivots, rhs, strict=True) if mask & (1 << i)) for i in range(len(checks)))


@dataclass(frozen=True)
class StateInjection:
    """Physical seed, projection, and measurement-dependent encoding byproducts.

    Render ``seed`` and ``projection`` with the existing gadget renderers.
    After reading the projection's X outcomes, apply ``correction_gadget``.
    A static TickCircuit cannot perform this classical feed-forward itself.
    """

    seed: Gadget
    projection: Gadget
    correction_supports: tuple[tuple[int, ...], ...]

    def correction_gadget(self, x_outcomes: tuple[bool, ...]) -> Gadget:
        """Materialize one observed branch; do not assume all outcomes are zero."""
        if len(x_outcomes) != len(self.correction_supports) or any(type(bit) is not bool for bit in x_outcomes):
            msg = "Expected one boolean outcome per X stabilizer"
            raise ValueError(msg)
        support: set[int] = set()
        for bit, correction in zip(x_outcomes, self.correction_supports, strict=True):
            if bit:
                support.symmetric_difference_update(correction)
        data = self.seed.allocations[0].data_qubits
        steps = tuple(SurfaceCircuitStep(OpType.Z, [data[q]]) for q in sorted(support))
        return Gadget(
            GadgetKind.TRANSVERSAL,
            "correct_injection_signs",
            steps,
            self.seed.allocations,
            self.seed.dimensions,
            "Z",
        )


def state_injection(
    patch: SurfacePatch,
    allocation: QubitAllocation | None = None,
    *,
    state: str = "T",
) -> StateInjection:
    """Encode Z/X/Y eigenstates or a raw T/TDG magic seed into a rotated patch.

    Supports square and rectangular rotated patches with dimensions >= 2.
    Prepare all data in zero, prepare the first logical-X site in ``state``,
    and spread it along that string with nearest-string-neighbour CX gates.
    The state is alpha|0...0> + beta X_L|0...0>, so all Z checks are +1.
    Project X checks, then remove their signs using logical-preserving Zs.
    States are Z, -Z, X, -X, Y, -Y, T, TDG (T means T|+>).

    There is no postselection, growth, distillation, or noisy decoding here.
    """
    if not patch.rotated or min(patch.dx, patch.dz) < 2:
        msg = "State injection requires a rotated patch with dx and dz >= 2"
        raise ValueError(msg)
    state = state.upper()
    gates = {
        "Z": (),
        "-Z": (OpType.X,),
        "X": (OpType.H,),
        "-X": (OpType.H, OpType.Z),
        "Y": (OpType.H, OpType.SZ),
        "-Y": (OpType.H, OpType.SZDG),
        "T": (OpType.H, OpType.T),
        "TDG": (OpType.H, OpType.TDG),
    }
    if state not in gates:
        msg = f"Unsupported injection state {state!r}; expected one of {tuple(gates)}"
        raise ValueError(msg)
    if allocation is None:
        allocation = default_allocation(patch)
    _validate_injection_allocation(patch, allocation)
    data = allocation.data_qubits
    logical_x = patch.geometry.logical_x.data_qubits
    steps = [SurfaceCircuitStep(OpType.ALLOC, [q], f"data[{i}]") for i, q in enumerate(data)]
    steps.append(SurfaceCircuitStep(OpType.TICK))
    for gate in gates[state]:
        steps.extend((SurfaceCircuitStep(gate, [data[logical_x[0]]]), SurfaceCircuitStep(OpType.TICK)))
    for control, target in pairwise(logical_x):
        steps.extend((SurfaceCircuitStep(OpType.CX, [data[control], data[target]]), SurfaceCircuitStep(OpType.TICK)))
    seed = Gadget(
        GadgetKind.PREP,
        f"prep_injection_{state.lower().replace('-', 'minus_')}_seed",
        tuple(steps),
        (allocation,),
        (patch.dx, patch.dz),
        state,
    )
    return StateInjection(
        seed,
        syndrome_round_gadget(patch, allocation, round_index=0),
        injection_correction_supports(patch),
    )


def _validate_injection_allocation(patch: SurfacePatch, allocation: QubitAllocation) -> None:
    """Require dedicated, disjoint registers with the expected geometry."""
    expected = default_allocation(patch)
    registers = (allocation.data_qubits, allocation.x_ancilla_qubits, allocation.z_ancilla_qubits)
    if tuple(map(len, registers)) != tuple(
        map(len, (expected.data_qubits, expected.x_ancilla_qubits, expected.z_ancilla_qubits)),
    ):
        msg = "State injection allocation must match the patch dimensions"
        raise ValueError(msg)
    all_qubits = [q for register in registers for q in register]
    if any(type(q) is not int or q < 0 for q in all_qubits) or len(set(all_qubits)) != len(all_qubits):
        msg = "State injection requires distinct nonnegative data and dedicated ancilla qubits"
        raise ValueError(msg)
