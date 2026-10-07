# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Postselected XY-plane hook injection into a rotated surface patch.

Schedule based on Craig Gidney, *Cleaner magic states with hook injection*,
arXiv:2302.12292, section 2, and its reference circuit (CC BY 4.0):
https://doi.org/10.5281/zenodo.7575030. Adapted to PECOS geometry, physical
gadgets, true T rotations, and explicit measurement-dependent byproducts.

The injection round grows the distance-two seed to the requested patch.
Verification stays at that size; subsequent growth and noisy decoding are
not implemented here. Undetectable single faults remain at the injection site.
"""

from dataclasses import dataclass

from pecos.qec.surface.circuit_builder import OpType, QubitAllocation, SurfaceCircuitStep
from pecos.qec.surface.gadgets import Gadget, GadgetKind, default_allocation
from pecos.qec.surface.injection import _validate_injection_allocation, injection_correction_supports
from pecos.qec.surface.patch import SurfacePatch

# Doubled coordinates, with columns increasing right and rows increasing down.
_UR, _UL, _DR, _DL = (1, -1), (-1, -1), (1, 1), (-1, 1)


@dataclass(frozen=True)
class HookInjection:
    """Physical preparation, injection, verification, and acceptance contract.

    Execute ``seed``, then ``injection``, then ``verification`` at least once.
    Reject nonzero predictable first checks or any subsequent syndrome change.
    Random first-round signs are encoding byproducts, not detection events.
    On success, ``correction_gadget`` canonicalizes these known signs while
    preserving both logical observables. It is not a noisy decoder.
    """

    seed: Gadget
    injection: Gadget
    verification: Gadget
    predictable_x: tuple[int, ...]
    predictable_z: tuple[int, ...]
    z_corrections: tuple[tuple[int, ...], ...]
    x_corrections: tuple[tuple[int, ...], ...]

    def _validate_syndrome(self, syndrome: tuple[tuple[bool, ...], tuple[bool, ...]]) -> None:
        if len(syndrome) != 2 or tuple(map(len, syndrome)) != (len(self.z_corrections), len(self.x_corrections)):
            msg = "Expected X and Z outcomes matching the patch stabilizers"
            raise ValueError(msg)
        if any(type(bit) is not bool for family in syndrome for bit in family):
            msg = "Syndrome outcomes must be booleans"
            raise ValueError(msg)

    def accepts(self, records: tuple[tuple[tuple[bool, ...], tuple[bool, ...]], ...]) -> bool:
        """Evaluate a complete attempt, including at least one verification round."""
        if len(records) < 2:
            msg = "Hook injection requires an injection round and at least one verification round"
            raise ValueError(msg)
        for record in records:
            self._validate_syndrome(record)
        first_x, first_z = records[0]
        predictable = [first_x[i] for i in self.predictable_x] + [first_z[i] for i in self.predictable_z]
        return not any(predictable) and all(record == records[0] for record in records[1:])

    def correction_gadget(self, syndrome: tuple[tuple[bool, ...], tuple[bool, ...]]) -> Gadget:
        """Materialize ideal byproduct corrections for an accepted baseline syndrome."""
        self._validate_syndrome(syndrome)
        steps = []
        data = self.seed.allocations[0].data_qubits
        for outcomes, supports, op in (
            (syndrome[0], self.z_corrections, OpType.Z),
            (syndrome[1], self.x_corrections, OpType.X),
        ):
            support: set[int] = set()
            for bit, correction in zip(outcomes, supports, strict=True):
                if bit:
                    support.symmetric_difference_update(correction)
            steps.extend(SurfaceCircuitStep(op, [data[q]]) for q in sorted(support))
        return Gadget(
            GadgetKind.TRANSVERSAL,
            "correct_hook_signs",
            tuple(steps),
            self.seed.allocations,
            self.seed.dimensions,
            "XZ",
        )


def hook_injection(
    patch: SurfacePatch,
    allocation: QubitAllocation | None = None,
    *,
    state: str = "T",
) -> HookInjection:
    """Build the published hook schedule for a square rotated patch, distance >= 2.

    ``state`` is T, TDG, X, -X, Y, or -Y. Only the midpoint ancilla rotation
    changes. T and TDG require a non-Clifford simulator. X/Y variants allow
    Clifford fault studies, but their fidelity is not a T-state certificate.
    The four CX layers and reversed verification order are deliberate.
    """
    if not patch.rotated or patch.dx != patch.dz or patch.dx < 2:
        msg = "Hook injection requires a square rotated patch of distance >= 2"
        raise ValueError(msg)
    state = state.upper()
    rotations = {"T": OpType.T, "TDG": OpType.TDG, "X": None, "-X": OpType.Z, "Y": OpType.SZ, "-Y": OpType.SZDG}
    if state not in rotations:
        msg = f"Unsupported hook state {state!r}; expected one of {tuple(rotations)}"
        raise ValueError(msg)
    if allocation is None:
        allocation = default_allocation(patch)
    _validate_injection_allocation(patch, allocation)
    d = patch.dx
    data = allocation.data_qubits
    positions = {q: (2 * (q % d), 2 * (q // d)) for q in range(d * d)}
    data_at = {pos: q for q, pos in positions.items()}
    initial_x = {q for q, (x, y) in positions.items() if x <= y or q == 1}
    tiles = []
    predictable: dict[str, list[int]] = {"X": [], "Z": []}
    for basis, checks, ancillas in (
        ("X", patch.geometry.x_stabilizers, allocation.x_ancilla_qubits),
        ("Z", patch.geometry.z_stabilizers, allocation.z_ancilla_qubits),
    ):
        for i, (check, ancilla) in enumerate(zip(checks, ancillas, strict=True)):
            support = check.data_qubits
            x = sum(positions[q][0] for q in support) // len(support)
            y = sum(positions[q][1] for q in support) // len(support)
            if len(support) == 2:
                if basis == "X":
                    y += -1 if y == 0 else 1
                else:
                    x += -1 if x == 0 else 1
            tiles.append((basis, i, ancilla, (x, y)))
            if all((q in initial_x) == (basis == "X") for q in support):
                predictable[basis].append(i)
    seed_steps = [SurfaceCircuitStep(OpType.ALLOC, [q], f"data[{i}]") for i, q in enumerate(data)]
    seed_steps.extend(SurfaceCircuitStep(OpType.H, [data[q]]) for q in sorted(initial_x))
    seed_steps.append(SurfaceCircuitStep(OpType.TICK))
    seed = Gadget(
        GadgetKind.PREP,
        "prep_injection_hook_seed",
        tuple(seed_steps),
        (allocation,),
        (d, d),
        state,
    )

    def make_round(*, inject: bool) -> Gadget:
        steps = [SurfaceCircuitStep(OpType.ALLOC, [a], f"a{b.lower()}{i}") for b, i, a, _ in tiles]
        steps.extend(SurfaceCircuitStep(OpType.H, [a]) for b, _, a, _ in tiles if b == "X")
        steps.append(SurfaceCircuitStep(OpType.TICK))
        for layer in range(4):
            for basis, _, ancilla, (x, y) in tiles:
                order = (_UR, _UL, _DR, _DL) if basis == "X" else (_UR, _DR, _UL, _DL)
                if inject and x == y:
                    order = (_UR, _UL, _DR, _DL)
                if not inject:
                    order = order[::-1]
                ox, oy = order[layer]
                q = data_at.get((x + ox, y + oy))
                if q is None:
                    continue
                # Omit first-layer CXs with a |0> control or |+> target.
                if inject and layer == 0 and ((basis == "X") == (q in initial_x)):
                    continue
                pair = [ancilla, data[q]] if basis == "X" else [data[q], ancilla]
                steps.append(SurfaceCircuitStep(OpType.CX, pair))
            steps.append(SurfaceCircuitStep(OpType.TICK))
            if inject and layer == 1 and rotations[state] is not None:
                hook_ancilla = next(a for b, _, a, pos in tiles if b == "Z" and pos == (1, 1))
                steps.extend((SurfaceCircuitStep(rotations[state], [hook_ancilla]), SurfaceCircuitStep(OpType.TICK)))
        steps.extend(SurfaceCircuitStep(OpType.H, [a]) for b, _, a, _ in tiles if b == "X")
        steps.append(SurfaceCircuitStep(OpType.TICK))
        steps.extend(SurfaceCircuitStep(OpType.MEASURE, [a], f"a{b.lower()}{i}") for b, i, a, _ in tiles)
        steps.append(SurfaceCircuitStep(OpType.TICK))
        name = f"hook_injection_{state.lower().replace('-', 'minus_')}" if inject else "hook_verification"
        return Gadget(GadgetKind.SYNDROME_ROUND, name, tuple(steps), (allocation,), (d, d), None)

    return HookInjection(
        seed,
        make_round(inject=True),
        make_round(inject=False),
        tuple(predictable["X"]),
        tuple(predictable["Z"]),
        injection_correction_supports(patch),
        injection_correction_supports(patch, basis="Z"),
    )
