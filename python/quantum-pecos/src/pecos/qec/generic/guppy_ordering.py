# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Explicit compiler-ordering cuts, separate from physical schedule timing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos.qec.generic.scheduled import ScheduledCircuit


class GuppyOrdering(Enum):
    """Ordering contracts for Guppy lowering; none assigns gate durations."""

    FUNCTIONAL = "functional"
    UNITARY_REGIONS = "unitary_regions"
    FIXED_LIFETIME = "fixed_lifetime"
    ALL_MOMENTS = "all_moments"


@dataclass(frozen=True)
class GuppyFence:
    """Join every incoming live wire to every outgoing wire at one source cut."""

    after_moment: int
    qubits: tuple[int, ...]

    @property
    def joins(self) -> tuple[tuple[int, ...], ...]:
        """Collect into an anchor, then distribute its complete dependency set."""
        if len(self.qubits) == 1:
            return (self.qubits,)
        anchor, *others = self.qubits
        # The last collected wire already depends on all inputs.
        return tuple((anchor, q) for q in (*others, *others[:-1]))


@dataclass(frozen=True)
class GuppyLifetimeFence:
    """Join live roles and retired measurement events through a classical token."""

    after_moment: int
    qubits: tuple[int, ...]
    measurements: tuple[int, ...]


@dataclass(frozen=True)
class GuppyOrderingPlan:
    """Requested source-order constraints, not a runtime timing certificate."""

    mode: GuppyOrdering
    num_moments: int
    fences: tuple[GuppyFence | GuppyLifetimeFence, ...]
    allocation_calls: int = 0

    def to_dict(self) -> dict:
        """Expose covered and uncovered cuts without implying timing equivalence."""
        covered = {fence.after_moment for fence in self.fences}
        rows = []
        for fence in self.fences:
            row = {"after_moment": fence.after_moment, "live_roles": list(fence.qubits)}
            if isinstance(fence, GuppyLifetimeFence):
                row.update({"measurement_records": list(fence.measurements), "runtime_barrier_calls": 1,
                            "dependency_helper_calls": 2 * len(fence.qubits) + len(fence.measurements)})
            else:
                row["runtime_barrier_calls"] = len(fence.joins)
            rows.append(row)
        return {
            "mode": self.mode.value,
            "fences": rows,
            "uncovered_cuts": [i for i in range(self.num_moments - 1) if i not in covered],
            "runtime_barrier_calls": sum(row["runtime_barrier_calls"] for row in rows),
            "dependency_helper_calls": sum(row.get("dependency_helper_calls", 0) for row in rows),
            "ordered_allocation_calls": self.allocation_calls,
            "scope": "Compiler data dependencies and runtime barrier requests; no duration or pulse-timing guarantee",
            "physical_schedule_verified": False,
            "noise_equivalence_verified": False,
        }

    def describe(self) -> str:
        """Summarize the requested contract and the boundary of its evidence."""
        data = self.to_dict()
        return (
            f"Guppy {self.mode.value}: {len(self.fences)}/{max(0, self.num_moments - 1)} moment cuts joined; "
            f"{data['runtime_barrier_calls']} runtime barrier calls\n"
            f"Dependency-only helpers: {data['dependency_helper_calls']}; "
            f"ordered allocations: {self.allocation_calls}\n"
            f"Uncovered cuts: {data['uncovered_cuts']}\n"
            "Physical timing and noise equivalence: not established"
        )


def plan_guppy_ordering(circuit: ScheduledCircuit, mode: GuppyOrdering) -> GuppyOrderingPlan:
    """Plan fences within fixed-live unitary regions or a whole fixed-lifetime circuit.

    R, M, and RELABEL delimit regions in UNITARY_REGIONS mode, consistently with
    the source scheduler. FIXED_LIFETIME accepts initial R on every role, then
    only unitaries/empty moments/RELABEL, then terminal M on every role. The latter
    rejects repeated ancilla lifetimes rather than silently weakening its scope.
    ALL_MOMENTS uses an opaque classical dependency through measurement and
    allocation, covering every cut even with no live qubits.
    """
    if not isinstance(mode, GuppyOrdering):
        msg = "ordering must be a GuppyOrdering"
        raise TypeError(msg)
    moments = circuit.moments
    if mode == GuppyOrdering.FIXED_LIFETIME:
        def all_roles(moment: tuple, gate: str) -> bool:
            return len(moment) == circuit.num_qubits and all(op.name == gate for op in moment)

        if (
            len(moments) < 2
            or not all_roles(moments[0], "R")
            or not all_roles(moments[-1], "M")
            or any(op.name in {"R", "M"} for moment in moments[1:-1] for op in moment)
        ):
            msg = "fixed_lifetime requires initial R on all roles, no intermediate R/M, and terminal M on all roles"
            raise ValueError(msg)
    fences, live = [], set()
    measurement = 0
    boundaries = [any(op.name in {"R", "M", "RELABEL"} for op in moment) for moment in moments]
    for index, moment in enumerate(moments):
        measured = []
        for op in moment:
            if op.name == "R":
                live.add(op.qubits[0])
            elif op.name == "M":
                live.remove(op.qubits[0])
                measured.append(measurement)
                measurement += 1
        if mode == GuppyOrdering.ALL_MOMENTS and index + 1 < len(moments):
            fences.append(GuppyLifetimeFence(index, tuple(sorted(live)), tuple(measured)))
        elif index + 1 < len(moments) and live and (
            mode == GuppyOrdering.FIXED_LIFETIME
            or (mode == GuppyOrdering.UNITARY_REGIONS and not boundaries[index] and not boundaries[index + 1])
        ):
            fences.append(GuppyFence(index, tuple(sorted(live))))
    allocations = circuit.num_measurements if mode == GuppyOrdering.ALL_MOMENTS else 0
    return GuppyOrderingPlan(mode, len(moments), tuple(fences), allocations)
