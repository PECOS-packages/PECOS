# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Abstract circuit builder for surface code experiments.

This module provides a unified way to generate surface code circuits
that can be rendered to multiple formats:
- Guppy source code
- Stim circuit format
- PECOS TickCircuit (with explicit tick boundaries, similar to Stim)
- PECOS DagCircuit

The circuit structure is defined once and rendered to each target,
ensuring consistency across representations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from pecos.qec.surface.patch import (
        LogicalDescriptor,
        Stabilizer,
        StabilizerDescriptor,
        SurfacePatch,
        SurfacePatchDescriptor,
    )
    from pecos.quantum import DagCircuit, TickCircuit, TickHandle


class SurfaceDetectorDescriptor(TypedDict):
    """Public detector descriptor derived from TickCircuit metadata."""

    id: int
    detector_id: int
    stabilizer_kind: str
    stabilizer_index: int
    round: int
    is_final_round: bool
    coords: list[int]
    records: list[int]
    stabilizer_is_boundary: bool
    stabilizer_region: str
    schedule_rounds: list[int]
    schedule_start_round: int | None
    schedule_end_round: int | None
    schedule_entries: list[dict[str, int | str]]
    data_qubits: list[int]
    data_qubit_positions: list[list[int]]
    weight: int


class SurfaceObservableDescriptor(TypedDict):
    """Public observable descriptor derived from TickCircuit metadata."""

    id: int
    observable_id: int
    basis: str
    records: list[int]
    logical_type: str
    data_qubits: list[int]
    data_qubit_positions: list[list[int]]
    weight: int
    support_axis: str


class SurfaceMemoryExperimentDescriptor(TypedDict):
    """Public bundle describing a surface-memory experiment."""

    patch: SurfacePatchDescriptor
    basis: str
    num_rounds: int
    ancilla_budget: int | None
    x_stabilizers: list[StabilizerDescriptor]
    z_stabilizers: list[StabilizerDescriptor]
    stabilizers: list[StabilizerDescriptor]
    logicals: list[LogicalDescriptor]
    detectors: list[SurfaceDetectorDescriptor]
    observables: list[SurfaceObservableDescriptor]


class OpType(Enum):
    """Circuit operation types."""

    # Qubit management
    ALLOC = auto()  # Allocate qubit
    PREP = auto()  # Prepare qubit in |0>

    # Single-qubit gates
    H = auto()  # Hadamard
    X = auto()  # Pauli X
    Z = auto()  # Pauli Z

    # Two-qubit gates
    CX = auto()  # CNOT

    # Measurement
    MEASURE = auto()  # Destructive measurement

    # Structural
    TICK = auto()  # Layer separator
    COMMENT = auto()  # Comment/annotation


@dataclass
class CircuitOp:
    """A circuit operation."""

    op_type: OpType
    qubits: list[int] = field(default_factory=list)
    label: str = ""  # For comments, variable names, etc.


@dataclass
class QubitAllocation:
    """Track qubit allocations by role."""

    data_qubits: list[int]
    x_ancilla_qubits: list[int]  # Indexed by stabilizer index
    z_ancilla_qubits: list[int]  # Indexed by stabilizer index

    @property
    def total(self) -> int:
        """Total number of qubits."""
        return len(set(self.data_qubits) | set(self.x_ancilla_qubits) | set(self.z_ancilla_qubits))


def _normalize_ancilla_budget(total_ancilla: int, ancilla_budget: int | None) -> int:
    """Clamp ancilla budget to the valid range for a patch."""
    if ancilla_budget is None:
        return total_ancilla

    if ancilla_budget < 1:
        msg = f"ancilla_budget must be >= 1, got {ancilla_budget}"
        raise ValueError(msg)

    return min(ancilla_budget, total_ancilla)


def _batched_stabilizers(
    patch: SurfacePatch,
    ancilla_budget: int,
) -> list[list[tuple[str, int]]]:
    """Partition stabilizers into ancilla-reuse batches.

    This mirrors the public Guppy batching order so the abstract circuit and
    its native DEMs match the actual low-ancilla circuit family.
    """
    geom = patch.geometry
    stabilizers = [("X", stab.index) for stab in geom.x_stabilizers]
    stabilizers.extend(("Z", stab.index) for stab in geom.z_stabilizers)
    # Sort key is load-bearing: it mirrors Guppy's stabilizer ordering (ascending
    # index, X before Z on ties). Batched DEMs are compared against Guppy output
    # shot-for-shot in the Selene parity tests, so any change here will diverge
    # from the low-ancilla reference family.
    stabilizers.sort(key=lambda stab: (stab[1], 0 if stab[0] == "X" else 1))

    return [stabilizers[start : start + ancilla_budget] for start in range(0, len(stabilizers), ancilla_budget)]


def build_surface_code_circuit(
    patch: SurfacePatch,
    num_rounds: int,
    basis: str = "Z",
    ancilla_budget: int | None = None,
) -> tuple[list[CircuitOp], QubitAllocation]:
    """Build abstract circuit operations for a surface code memory experiment.

    This generates the circuit structure matching the Guppy implementation:
    1. prep_{basis}_basis: Allocate and prepare data qubits
    2. syndrome_extraction x num_rounds: Syndrome extraction with fresh ancillas
    3. measure_{basis}_basis: Final data qubit measurement

    Args:
        patch: Surface code patch with geometry
        num_rounds: Number of syndrome extraction rounds
        basis: 'Z' for |0_L> state or 'X' for |+_L> state
        ancilla_budget: Optional cap on simultaneously live ancillas. When
            provided below the total stabilizer count, ancillas are reused
            across stabilizer batches following the public Guppy order.

    Returns:
        Tuple of (operations list, qubit allocation info)
    """
    from pecos.qec.surface.schedule import compute_cnot_schedule

    geom = patch.geometry
    num_data = geom.num_data
    num_x_anc = len(geom.x_stabilizers)
    num_z_anc = len(geom.z_stabilizers)
    total_ancilla = num_x_anc + num_z_anc
    effective_ancilla_budget = _normalize_ancilla_budget(total_ancilla, ancilla_budget)

    # Qubit allocation layout. Under ancilla reuse, stabilizers map onto a
    # shared ancilla pool and different stabilizers can intentionally share the
    # same physical qubit id at different times.
    if effective_ancilla_budget == total_ancilla:
        allocation = QubitAllocation(
            data_qubits=list(range(num_data)),
            x_ancilla_qubits=list(range(num_data, num_data + num_x_anc)),
            z_ancilla_qubits=list(
                range(num_data + num_x_anc, num_data + num_x_anc + num_z_anc),
            ),
        )
    else:
        ancilla_pool = list(range(num_data, num_data + effective_ancilla_budget))
        x_ancilla_qubits = [-1] * num_x_anc
        z_ancilla_qubits = [-1] * num_z_anc
        for batch in _batched_stabilizers(patch, effective_ancilla_budget):
            for pool_idx, (stab_type, stab_idx) in enumerate(batch):
                if stab_type == "X":
                    x_ancilla_qubits[stab_idx] = ancilla_pool[pool_idx]
                else:
                    z_ancilla_qubits[stab_idx] = ancilla_pool[pool_idx]

        allocation = QubitAllocation(
            data_qubits=list(range(num_data)),
            x_ancilla_qubits=x_ancilla_qubits,
            z_ancilla_qubits=z_ancilla_qubits,
        )

    def data_q(i: int) -> int:
        return allocation.data_qubits[i]

    def x_anc_q(stab_idx: int) -> int:
        return allocation.x_ancilla_qubits[stab_idx]

    def z_anc_q(stab_idx: int) -> int:
        return allocation.z_ancilla_qubits[stab_idx]

    # Get CNOT schedule
    cnot_rounds = compute_cnot_schedule(patch)

    ops: list[CircuitOp] = []

    # =========================================================================
    # prep_z_basis / prep_x_basis
    # =========================================================================
    ops.append(CircuitOp(OpType.COMMENT, label=f"prep_{basis.lower()}_basis"))

    # Allocate and reset data qubits
    ops.extend(CircuitOp(OpType.ALLOC, [data_q(i)], f"data[{i}]") for i in range(num_data))

    # For X-basis: H on each data qubit
    if basis.upper() == "X":
        ops.extend(CircuitOp(OpType.H, [data_q(i)]) for i in range(num_data))

    ops.append(CircuitOp(OpType.TICK))

    # =========================================================================
    # syndrome_extraction (called num_rounds times)
    # =========================================================================
    for rnd in range(num_rounds):
        ops.append(
            CircuitOp(OpType.COMMENT, label=f"syndrome_extraction round {rnd + 1}"),
        )
        if effective_ancilla_budget == total_ancilla:
            ops.extend(CircuitOp(OpType.ALLOC, [x_anc_q(s.index)], f"ax{s.index}") for s in geom.x_stabilizers)
            ops.extend(CircuitOp(OpType.ALLOC, [z_anc_q(s.index)], f"az{s.index}") for s in geom.z_stabilizers)

            ops.append(CircuitOp(OpType.COMMENT, label="Hadamard on X ancillas"))
            ops.extend(CircuitOp(OpType.H, [x_anc_q(s.index)], f"ax{s.index}") for s in geom.x_stabilizers)

            ops.append(CircuitOp(OpType.TICK))

            for rnd_idx, cx_round in enumerate(cnot_rounds):
                ops.append(CircuitOp(OpType.COMMENT, label=f"CX round {rnd_idx + 1}"))
                for stab_type, stab_idx, data_idx in cx_round:
                    if stab_type == "X":
                        ops.append(
                            CircuitOp(
                                OpType.CX,
                                [x_anc_q(stab_idx), data_q(data_idx)],
                                f"X{stab_idx}",
                            ),
                        )
                    else:
                        ops.append(
                            CircuitOp(
                                OpType.CX,
                                [data_q(data_idx), z_anc_q(stab_idx)],
                                f"Z{stab_idx}",
                            ),
                        )
                ops.append(CircuitOp(OpType.TICK))

            ops.append(CircuitOp(OpType.COMMENT, label="Hadamard on X ancillas"))
            ops.extend(CircuitOp(OpType.H, [x_anc_q(s.index)], f"ax{s.index}") for s in geom.x_stabilizers)

            ops.append(CircuitOp(OpType.COMMENT, label="Measure ancillas"))
            ops.extend(CircuitOp(OpType.MEASURE, [x_anc_q(s.index)], f"sx{s.index}") for s in geom.x_stabilizers)
            ops.extend(CircuitOp(OpType.MEASURE, [z_anc_q(s.index)], f"sz{s.index}") for s in geom.z_stabilizers)

            ops.append(CircuitOp(OpType.TICK))
        else:
            stabilizer_batches = _batched_stabilizers(patch, effective_ancilla_budget)
            for batch in stabilizer_batches:
                ops.append(CircuitOp(OpType.COMMENT, label="Prepare ancillas"))
                batch_ancillas = {
                    (stab_type, stab_idx): x_anc_q(stab_idx) if stab_type == "X" else z_anc_q(stab_idx)
                    for stab_type, stab_idx in batch
                }

                for stab_type, stab_idx in batch:
                    ops.append(
                        CircuitOp(
                            OpType.ALLOC,
                            [batch_ancillas[(stab_type, stab_idx)]],
                            f"a{stab_type.lower()}{stab_idx}",
                        ),
                    )

                x_stabilizers_in_batch = [stab_idx for stab_type, stab_idx in batch if stab_type == "X"]
                if x_stabilizers_in_batch:
                    ops.append(CircuitOp(OpType.COMMENT, label="Hadamard on X ancillas"))
                    ops.extend(
                        CircuitOp(OpType.H, [batch_ancillas[("X", stab_idx)]], f"ax{stab_idx}")
                        for stab_idx in x_stabilizers_in_batch
                    )

                ops.append(CircuitOp(OpType.TICK))

                for rnd_idx, cx_round in enumerate(cnot_rounds):
                    ops.append(CircuitOp(OpType.COMMENT, label=f"CX round {rnd_idx + 1}"))
                    for stab_type, stab_idx, data_idx in cx_round:
                        ancilla_q = batch_ancillas.get((stab_type, stab_idx))
                        if ancilla_q is None:
                            continue
                        if stab_type == "X":
                            ops.append(
                                CircuitOp(
                                    OpType.CX,
                                    [ancilla_q, data_q(data_idx)],
                                    f"X{stab_idx}",
                                ),
                            )
                        else:
                            ops.append(
                                CircuitOp(
                                    OpType.CX,
                                    [data_q(data_idx), ancilla_q],
                                    f"Z{stab_idx}",
                                ),
                            )
                    ops.append(CircuitOp(OpType.TICK))

                if x_stabilizers_in_batch:
                    ops.append(CircuitOp(OpType.COMMENT, label="Hadamard on X ancillas"))
                    ops.extend(
                        CircuitOp(OpType.H, [batch_ancillas[("X", stab_idx)]], f"ax{stab_idx}")
                        for stab_idx in x_stabilizers_in_batch
                    )

                ops.append(CircuitOp(OpType.COMMENT, label="Measure ancillas"))
                for stab_type, stab_idx in batch:
                    measure_label = f"sx{stab_idx}" if stab_type == "X" else f"sz{stab_idx}"
                    ops.append(
                        CircuitOp(
                            OpType.MEASURE,
                            [batch_ancillas[(stab_type, stab_idx)]],
                            measure_label,
                        ),
                    )

                ops.append(CircuitOp(OpType.TICK))

    # =========================================================================
    # measure_z_basis / measure_x_basis
    # =========================================================================
    ops.append(CircuitOp(OpType.COMMENT, label=f"measure_{basis.lower()}_basis"))

    # For X-basis: H on each data qubit first
    if basis.upper() == "X":
        ops.extend(CircuitOp(OpType.H, [data_q(i)]) for i in range(num_data))

    # Measure all data qubits
    ops.extend(CircuitOp(OpType.MEASURE, [data_q(i)], f"final[{i}]") for i in range(num_data))

    return ops, allocation


def classify_stabilizer_boundary(stab_type: str, data_qubits: tuple[int, ...], d: int) -> str:
    """Public wrapper for classifying a boundary stabilizer."""
    from pecos.qec.surface.schedule import _classify_boundary

    return _classify_boundary(stab_type, data_qubits, d)


def get_stabilizer_region(stab: Stabilizer, patch: SurfacePatch) -> str:
    """Return a coarse region label like ``top+left`` for a stabilizer."""
    geom = patch.geometry
    positions = [geom.id_to_pos[q] for q in stab.data_qubits]
    avg_row = sum(row for row, _ in positions) / len(positions)
    avg_col = sum(col for _, col in positions) / len(positions)
    row_label = "top" if avg_row < (geom.dx - 1) / 2 else "bottom"
    col_label = "left" if avg_col < (geom.dz - 1) / 2 else "right"
    return f"{row_label}+{col_label}"


def get_stabilizer_touch_label(stab: Stabilizer, patch: SurfacePatch, data_qubit: int) -> str:
    """Label how a data qubit sits relative to a stabilizer support."""
    geom = patch.geometry
    if data_qubit not in stab.data_qubits:
        msg = f"Qubit {data_qubit} is not in stabilizer {stab.stab_type}{stab.index}"
        raise ValueError(msg)

    positions = [geom.id_to_pos[q] for q in stab.data_qubits]
    data_row, data_col = geom.id_to_pos[data_qubit]
    rows = [row for row, _ in positions]
    cols = [col for _, col in positions]

    if len(set(rows)) == 1:
        return "left" if data_col == min(cols) else "right"
    if len(set(cols)) == 1:
        return "top" if data_row == min(rows) else "bottom"

    vertical = "T" if data_row == min(rows) else "B"
    horizontal = "L" if data_col == min(cols) else "R"
    return vertical + horizontal


def get_stabilizer_schedule_entries(stab: Stabilizer, patch: SurfacePatch) -> list[dict[str, int | str]]:
    """Return the per-round touch schedule for one stabilizer."""
    from pecos.qec.surface.schedule import get_stab_schedule

    schedule = get_stab_schedule(stab.stab_type, stab.data_qubits, stab.is_boundary, patch.distance)
    return [
        {
            "round_0based": round_0based,
            "data_qubit": data_qubit,
            "touch_label": get_stabilizer_touch_label(stab, patch, data_qubit),
        }
        for round_0based, data_qubit in schedule
    ]


def get_stabilizer_schedule_metadata(stab: Stabilizer, patch: SurfacePatch) -> dict[str, object]:
    """Return metadata describing one stabilizer's schedule and geometry."""
    entries = get_stabilizer_schedule_entries(stab, patch)
    rounds = [int(entry["round_0based"]) for entry in entries]
    return {
        "stabilizer_kind": stab.stab_type,
        "stabilizer_index": stab.index,
        "stabilizer_is_boundary": stab.is_boundary,
        "stabilizer_region": get_stabilizer_region(stab, patch),
        "schedule_rounds": rounds,
        "schedule_start_round": rounds[0] if rounds else None,
        "schedule_end_round": rounds[-1] if rounds else None,
        "schedule_entries": entries,
    }


def _build_detector_descriptors(
    detectors: list[dict[str, object]],
    patch: SurfacePatch,
) -> list[SurfaceDetectorDescriptor]:
    """Build enriched detector descriptors from TickCircuit detector metadata."""
    num_x_anc = len(patch.x_stabilizers)
    final_round = max((int(det["coords"][2]) for det in detectors), default=-1)
    descriptors: list[SurfaceDetectorDescriptor] = []

    for det in detectors:
        coords = [int(value) for value in det["coords"]]
        records = [int(value) for value in det["records"]]
        raw_index = coords[0]
        if coords[1] == 0:
            stab_kind = "X"
            stab_index = raw_index
        else:
            stab_kind = "Z"
            stab_index = raw_index - num_x_anc

        descriptor = patch.get_stabilizer_descriptor(stab_kind, stab_index)
        descriptors.append(
            {
                "id": int(det["id"]),
                "detector_id": int(det["id"]),
                "stabilizer_kind": descriptor["stabilizer_kind"],
                "stabilizer_index": descriptor["stabilizer_index"],
                "round": coords[2],
                "is_final_round": coords[2] == final_round,
                "coords": coords,
                "records": records,
                "stabilizer_is_boundary": descriptor["stabilizer_is_boundary"],
                "stabilizer_region": descriptor["stabilizer_region"],
                "schedule_rounds": descriptor["schedule_rounds"],
                "schedule_start_round": descriptor["schedule_start_round"],
                "schedule_end_round": descriptor["schedule_end_round"],
                "schedule_entries": descriptor["schedule_entries"],
                "data_qubits": descriptor["data_qubits"],
                "data_qubit_positions": descriptor["data_qubit_positions"],
                "weight": descriptor["weight"],
            },
        )

    return descriptors


def _build_observable_descriptors(
    observables: list[dict[str, object]],
    patch: SurfacePatch,
    basis: str,
) -> list[SurfaceObservableDescriptor]:
    """Build enriched logical observable descriptors from TickCircuit metadata."""
    logical = patch.get_logical_descriptor(basis.upper())
    return [
        {
            "id": int(obs["id"]),
            "observable_id": int(obs["id"]),
            "basis": basis.upper(),
            "records": [int(value) for value in obs["records"]],
            "logical_type": logical["logical_type"],
            "data_qubits": logical["data_qubits"],
            "data_qubit_positions": logical["data_qubit_positions"],
            "weight": logical["weight"],
            "support_axis": logical["support_axis"],
        }
        for obs in observables
    ]


class CircuitRenderer(ABC):
    """Abstract base class for circuit renderers."""

    @abstractmethod
    def render(
        self,
        ops: list[CircuitOp],
        allocation: QubitAllocation,
        patch: SurfacePatch,
        num_rounds: int,
        basis: str,
    ) -> str:
        """Render operations to target format."""


class StimRenderer(CircuitRenderer):
    """Render circuit operations to Stim format."""

    def __init__(
        self,
        *,
        p1: float = 0.0,
        p2: float = 0.0,
        p_meas: float = 0.0,
        p_init: float = 0.0,
        add_detectors: bool = True,
    ) -> None:
        """Initialize Stim renderer.

        Args:
            p1: Single-qubit depolarizing error rate
            p2: Two-qubit depolarizing error rate
            p_meas: Measurement error rate
            p_init: Initialization error rate
            add_detectors: Whether to add DETECTOR annotations
        """
        self.p1 = p1
        self.p2 = p2
        self.p_meas = p_meas
        self.p_init = p_init
        self.add_detectors = add_detectors

    def render(
        self,
        ops: list[CircuitOp],
        allocation: QubitAllocation,
        patch: SurfacePatch,
        num_rounds: int,
        basis: str,
    ) -> str:
        """Render to Stim circuit string."""
        geom = patch.geometry
        num_x_anc = len(geom.x_stabilizers)

        lines = []
        lines.append(
            f"# Surface code d={patch.distance} {basis}-basis memory experiment",
        )
        lines.append(f"# {num_rounds} syndrome rounds, {allocation.total} qubits")
        lines.append("")

        # Track measurements for detector annotations
        meas_count = 0
        stab_meas_record: dict[tuple[str, int, int], int] = {}
        current_round = -1  # Track syndrome round
        final_meas_start = 0

        for op in ops:
            if op.op_type == OpType.COMMENT:
                if "syndrome_extraction round" in op.label:
                    current_round = int(op.label.split()[-1]) - 1
                lines.append(f"# {op.label}")

            elif op.op_type == OpType.ALLOC:
                lines.append(f"R {op.qubits[0]}")
                if self.p_init > 0:
                    lines.append(f"X_ERROR({self.p_init}) {op.qubits[0]}")

            elif op.op_type == OpType.H:
                lines.append(f"H {op.qubits[0]}")
                if self.p1 > 0:
                    lines.append(f"DEPOLARIZE1({self.p1}) {op.qubits[0]}")

            elif op.op_type == OpType.CX:
                c, t = op.qubits
                lines.append(f"CX {c} {t}")
                if self.p2 > 0:
                    lines.append(f"DEPOLARIZE2({self.p2}) {c} {t}")

            elif op.op_type == OpType.MEASURE:
                q = op.qubits[0]
                if self.p_meas > 0:
                    lines.append(f"X_ERROR({self.p_meas}) {q}")
                lines.append(f"M {q}")

                # Track measurement index
                if op.label.startswith("sx"):
                    stab_idx = int(op.label[2:])
                    stab_meas_record[("X", stab_idx, current_round)] = meas_count
                elif op.label.startswith("sz"):
                    stab_idx = int(op.label[2:])
                    stab_meas_record[("Z", stab_idx, current_round)] = meas_count
                elif op.label.startswith("final"):
                    if "final[0]" in op.label:
                        final_meas_start = meas_count
                meas_count += 1

            elif op.op_type == OpType.TICK:
                lines.append("TICK")

        # Add detector annotations if requested
        if self.add_detectors:
            lines.append("")
            lines.append("# Detectors")

            # Determine which stabilizer types are deterministic in round 0
            # Z-basis: Z stabilizers are deterministic (eigenvalue +1 on |0>)
            # X-basis: X stabilizers are deterministic (eigenvalue +1 on |+>)
            deterministic_type_round0 = "Z" if basis.upper() == "Z" else "X"

            # Syndrome detectors for X stabilizers
            for rnd in range(num_rounds):
                for s in geom.x_stabilizers:
                    curr_idx = stab_meas_record.get(("X", s.index, rnd))
                    if curr_idx is None:
                        continue
                    curr_offset = meas_count - curr_idx

                    if rnd == 0:
                        # Only X stabilizers have deterministic round-0 detectors in X-basis
                        if deterministic_type_round0 == "X":
                            lines.append(
                                f"DETECTOR({s.index}, 0, {rnd}) rec[{-curr_offset}]",
                            )
                        # In Z-basis, X stabilizers are random in round 0, skip single-record detector
                    else:
                        # Compare consecutive rounds (always valid)
                        prev_idx = stab_meas_record[("X", s.index, rnd - 1)]
                        prev_offset = meas_count - prev_idx
                        lines.append(
                            f"DETECTOR({s.index}, 0, {rnd}) rec[{-curr_offset}] rec[{-prev_offset}]",
                        )

            # Syndrome detectors for Z stabilizers
            for rnd in range(num_rounds):
                for s in geom.z_stabilizers:
                    curr_idx = stab_meas_record.get(("Z", s.index, rnd))
                    if curr_idx is None:
                        continue
                    curr_offset = meas_count - curr_idx
                    det_x = num_x_anc + s.index

                    if rnd == 0:
                        # Only Z stabilizers have deterministic round-0 detectors in Z-basis
                        if deterministic_type_round0 == "Z":
                            lines.append(
                                f"DETECTOR({det_x}, 1, {rnd}) rec[{-curr_offset}]",
                            )
                        # In X-basis, Z stabilizers are random in round 0, skip single-record detector
                    else:
                        # Compare consecutive rounds (always valid)
                        prev_idx = stab_meas_record[("Z", s.index, rnd - 1)]
                        prev_offset = meas_count - prev_idx
                        lines.append(
                            f"DETECTOR({det_x}, 1, {rnd}) rec[{-curr_offset}] rec[{-prev_offset}]",
                        )

            # Final detectors: compare last syndrome measurement to final data measurement
            # Only for stabilizers that match the measurement basis
            if basis.upper() == "Z":
                stabilizers = geom.z_stabilizers
                stab_type = "Z"
                logical_qubits = list(geom.logical_z.data_qubits) if geom.logical_z else []
            else:
                stabilizers = geom.x_stabilizers
                stab_type = "X"
                logical_qubits = list(geom.logical_x.data_qubits) if geom.logical_x else []

            for s in stabilizers:
                data_rec_offsets = [meas_count - (final_meas_start + dq) for dq in s.data_qubits]
                last_syn_idx = stab_meas_record[(stab_type, s.index, num_rounds - 1)]
                syn_offset = meas_count - last_syn_idx
                rec_str = " ".join(f"rec[{-off}]" for off in data_rec_offsets)
                det_x = s.index if stab_type == "X" else num_x_anc + s.index
                det_y = 0 if stab_type == "X" else 1
                lines.append(
                    f"DETECTOR({det_x}, {det_y}, {num_rounds}) {rec_str} rec[{-syn_offset}]",
                )

            # Logical observable
            logical_rec_offsets = [meas_count - (final_meas_start + q) for q in logical_qubits]
            logical_rec_str = " ".join(f"rec[{-off}]" for off in logical_rec_offsets)
            lines.append(f"OBSERVABLE_INCLUDE(0) {logical_rec_str}")

        return "\n".join(lines)


class GuppyRenderer(CircuitRenderer):
    """Render circuit operations to Guppy source code.

    This renderer produces the same modular Guppy code structure as
    pecos.guppy.surface.generate_guppy_source(), ensuring consistency.
    """

    def render(
        self,
        _ops: list[CircuitOp],
        _allocation: QubitAllocation,
        patch: SurfacePatch,
        _num_rounds: int,
        _basis: str,
    ) -> str:
        """Render to Guppy source code.

        Generates a full Guppy module with:
        - Struct definitions (SurfaceCode, Syndrome)
        - State preparation functions (prep_z_basis, prep_x_basis)
        - Syndrome extraction function
        - Measurement functions
        - Logical operator functions
        - Memory experiment factories (make_memory_z, make_memory_x)
        """
        from pecos.guppy.surface import generate_guppy_source

        # Use the canonical Guppy generator to ensure identical output
        return generate_guppy_source(patch)


class DagCircuitRenderer(CircuitRenderer):
    """Render circuit operations to PECOS DagCircuit."""

    def render(
        self,
        ops: list[CircuitOp],
        _allocation: QubitAllocation,
        _patch: SurfacePatch,
        _num_rounds: int,
        _basis: str,
    ) -> DagCircuit:
        """Render to PECOS DagCircuit."""
        from pecos_rslib import DagCircuit

        circuit = DagCircuit()
        allocated: set[int] = set()

        for op in ops:
            if op.op_type == OpType.COMMENT:
                pass  # DagCircuit doesn't support comments

            elif op.op_type == OpType.ALLOC:
                q = op.qubits[0]
                if q not in allocated:
                    circuit.qalloc([q])
                    allocated.add(q)
                else:
                    # Re-allocation acts as reset - use pz (prep Z / reset)
                    circuit.pz([q])

            elif op.op_type == OpType.PREP:
                circuit.pz([op.qubits[0]])

            elif op.op_type == OpType.H:
                circuit.h([op.qubits[0]])

            elif op.op_type == OpType.X:
                circuit.x([op.qubits[0]])

            elif op.op_type == OpType.Z:
                circuit.z([op.qubits[0]])

            elif op.op_type == OpType.CX:
                circuit.cx([(op.qubits[0], op.qubits[1])])

            elif op.op_type == OpType.MEASURE:
                circuit.mz([op.qubits[0]])

            elif op.op_type == OpType.TICK:
                pass  # DagCircuit doesn't have explicit ticks

        return circuit


class TickCircuitRenderer(CircuitRenderer):
    """Render circuit operations to PECOS TickCircuit.

    TickCircuit has explicit tick boundaries similar to Stim's TICK instruction.
    Operations within a tick run in parallel (no qubit conflicts allowed).
    This provides a 1:1 correspondence with Stim's tick structure.

    When qubit conflicts occur within a tick (same qubit used twice),
    a new tick is automatically created to maintain valid parallel structure.

    Detector annotations (similar to Stim's DETECTOR and OBSERVABLE_INCLUDE)
    are stored as circuit metadata and preserved when converting to DagCircuit.
    """

    def __init__(self, *, add_detectors: bool = True) -> None:
        """Initialize TickCircuit renderer.

        Args:
            add_detectors: Whether to add detector annotations as metadata
        """
        self.add_detectors = add_detectors

    def render(
        self,
        ops: list[CircuitOp],
        allocation: QubitAllocation,
        patch: SurfacePatch,
        num_rounds: int,
        basis: str,
    ) -> TickCircuit:
        """Render to PECOS TickCircuit.

        The tick structure follows Stim's pattern:
        - Tick: Prep data qubits
        - Tick: H for X-basis prep (if X-basis)
        - For each syndrome round:
            - Tick: Prep ancillas
            - Tick: H on X ancillas
            - Tick: CX round 1
            - Tick: CX round 2
            - Tick: CX round 3
            - Tick: CX round 4
            - Tick: H on X ancillas
            - Tick: Measure ancillas
        - Tick: H for X-basis measure (if X-basis)
        - Tick: Measure data qubits

        Metadata is stored at three levels:
        - Circuit-level (preserved in DagCircuit):
            - 'detectors': JSON list of {id, coords, records}
            - 'observables': JSON list of {id, records}
            - 'num_measurements', 'num_detectors', 'basis'
        - Tick-level: 'phase', 'syndrome_round', 'cx_round'
        - Gate-level: 'label', 'role'
        """
        import json

        from pecos_rslib.quantum import TickCircuit

        circuit = TickCircuit()
        geom = patch.geometry
        allocated: set[int] = set()
        current_tick_handle = None
        current_tick_idx = -1
        qubits_in_current_tick: set[int] = set()

        # Track measurements for detector annotations
        meas_count = 0
        stab_meas_record: dict[tuple[str, int, int], int] = {}
        current_round = -1
        current_phase = "prep"
        current_cx_round = 0
        final_meas_start = 0

        # Store all tick metadata to apply at the end (workaround for metadata
        # being lost when new ticks are created)
        # Format: {tick_idx: {'phase': str, 'round': int, 'cx_round': int, 'gates': [(label, role), ...]}}
        all_tick_metadata: dict[int, dict] = {}

        def get_stabilizer_from_label(label: str) -> str:
            """Decode surface stabilizer identity from an operation label."""
            if not label:
                return ""
            if label[0] in {"X", "Z"} and label[1:].isdigit():
                return label
            if label.startswith(("ax", "sx")) and label[2:].isdigit():
                return f"X{int(label[2:])}"
            if label.startswith(("az", "sz")) and label[2:].isdigit():
                return f"Z{int(label[2:])}"
            return ""

        # Helper to get stabilizer name for a CX gate
        def get_cx_stabilizer(control: int, target: int, label: str = "") -> str:
            """Get stabilizer name for a CX gate (e.g., 'X0', 'Z2')."""
            from_label = get_stabilizer_from_label(label)
            if from_label:
                return from_label
            if control in allocation.x_ancilla_qubits:
                # X stabilizer: ancilla is control
                stab_idx = allocation.x_ancilla_qubits.index(control)
                return f"X{stab_idx}"
            if target in allocation.z_ancilla_qubits:
                # Z stabilizer: ancilla is target
                stab_idx = allocation.z_ancilla_qubits.index(target)
                return f"Z{stab_idx}"
            return ""

        stabilizer_by_label = {
            **{f"X{s.index}": s for s in geom.x_stabilizers},
            **{f"Z{s.index}": s for s in geom.z_stabilizers},
        }
        stabilizer_by_ancilla_qubit = {
            **{allocation.x_ancilla_qubits[s.index]: f"X{s.index}" for s in geom.x_stabilizers},
            **{allocation.z_ancilla_qubits[s.index]: f"Z{s.index}" for s in geom.z_stabilizers},
        }

        def get_stabilizer_metadata(stab_label: str) -> dict[str, object]:
            stab = stabilizer_by_label[stab_label]
            return {
                "stabilizer": stab_label,
                "stabilizer_kind": stab.stab_type,
                "stabilizer_index": stab.index,
                "stabilizer_is_boundary": stab.is_boundary,
                "stabilizer_region": get_stabilizer_region(stab, patch),
            }

        def get_ancilla_gate_metadata(qubit: int, label: str = "") -> dict[str, object]:
            stab_label = get_stabilizer_from_label(label) or stabilizer_by_ancilla_qubit.get(qubit)
            if stab_label is None:
                return {}
            metadata = get_stabilizer_metadata(stab_label)
            metadata["ancilla_qubit"] = qubit
            return metadata

        def get_cx_gate_metadata(control: int, target: int, label: str = "") -> dict[str, object]:
            stab_label = get_cx_stabilizer(control, target, label)
            if not stab_label:
                return {}
            metadata = get_stabilizer_metadata(stab_label)
            ancilla_qubit = next(
                (q for q in (control, target) if q in stabilizer_by_ancilla_qubit),
                None,
            )
            data_qubit = next((q for q in (control, target) if q in allocation.data_qubits), None)
            if ancilla_qubit is not None:
                metadata["ancilla_qubit"] = ancilla_qubit
            if data_qubit is not None:
                metadata["data_qubit"] = data_qubit
                metadata["touch_label"] = get_stabilizer_touch_label(
                    stabilizer_by_label[stab_label],
                    patch,
                    data_qubit,
                )
            if current_cx_round > 0:
                metadata["cx_round_0based"] = current_cx_round - 1
            return metadata

        def new_tick() -> TickHandle:
            nonlocal current_tick_handle, current_tick_idx, qubits_in_current_tick
            current_tick_handle = circuit.tick()
            # Use next_tick_index() - 1 instead of num_ticks() - 1 because
            # num_ticks() excludes trailing empty ticks
            current_tick_idx = circuit.next_tick_index() - 1
            qubits_in_current_tick = set()
            # Initialize metadata storage for this tick
            all_tick_metadata[current_tick_idx] = {
                "phase": current_phase,
                "round": current_round,
                "cx_round": current_cx_round,
                "gates": [],
            }
            return current_tick_handle

        def ensure_tick() -> TickHandle:
            if current_tick_handle is None:
                return new_tick()
            return current_tick_handle

        def get_tick_for_qubits(qubits: list[int]) -> TickHandle:
            """Get a tick that can accept these qubits (no conflicts)."""
            if qubits_in_current_tick & set(qubits):
                return new_tick()
            return ensure_tick()

        def mark_qubits_used(qubits: list[int]) -> None:
            """Mark qubits as used in current tick."""
            qubits_in_current_tick.update(qubits)

        def queue_gate_metadata(meta: dict | None = None) -> None:
            """Queue metadata for the current gate.

            Args:
                meta: Optional dict with gate metadata (e.g., {"label": "data[0]"})
            """
            if current_tick_idx >= 0:
                context = {
                    "phase": current_phase,
                }
                if current_round >= 0:
                    context["syndrome_round"] = current_round
                if current_cx_round > 0:
                    context["cx_round"] = current_cx_round
                merged_meta = context
                if meta:
                    merged_meta = {**context, **meta}
                all_tick_metadata[current_tick_idx]["gates"].append(merged_meta)

        for op in ops:
            if op.op_type == OpType.COMMENT:
                # Track phase from comments
                if "syndrome_extraction round" in op.label:
                    current_round = int(op.label.split()[-1]) - 1
                    current_phase = "syndrome_prep"
                    current_cx_round = 0
                elif "Prepare ancillas" in op.label:
                    current_phase = "syndrome_prep"
                    current_cx_round = 0
                elif "Hadamard on X ancillas" in op.label:
                    current_phase = "syndrome_h_pre" if current_phase == "syndrome_prep" else "syndrome_h_post"
                elif "CX round" in op.label:
                    current_cx_round = int(op.label.split()[-1])
                    current_phase = f"cx_round_{current_cx_round}"
                elif "Measure ancillas" in op.label:
                    current_phase = "measure_ancilla"
                elif "prep_z_basis" in op.label or "prep_x_basis" in op.label:
                    current_phase = "prep_data"
                elif "measure_z_basis" in op.label or "measure_x_basis" in op.label:
                    current_phase = "measure_data"

            elif op.op_type == OpType.ALLOC:
                q = op.qubits[0]
                tick = get_tick_for_qubits([q])
                if q not in allocated:
                    tick.qalloc([q])
                    allocated.add(q)
                else:
                    tick.pz([q])
                mark_qubits_used([q])
                # Label helps identify which qubit (e.g., "data[0]", "ax0")
                meta = get_ancilla_gate_metadata(q, op.label)
                if op.label:
                    meta["label"] = op.label
                queue_gate_metadata(meta or None)

            elif op.op_type == OpType.PREP:
                q = op.qubits[0]
                get_tick_for_qubits([q]).pz([q])
                mark_qubits_used([q])
                meta = get_ancilla_gate_metadata(q, op.label)
                if op.label:
                    meta["label"] = op.label
                queue_gate_metadata(meta or None)

            elif op.op_type == OpType.H:
                q = op.qubits[0]
                get_tick_for_qubits([q]).h([q])
                mark_qubits_used([q])
                meta = get_ancilla_gate_metadata(q, op.label)
                if op.label:
                    meta["label"] = op.label
                queue_gate_metadata(meta or None)

            elif op.op_type == OpType.X:
                q = op.qubits[0]
                get_tick_for_qubits([q]).x([q])
                mark_qubits_used([q])
                meta = get_ancilla_gate_metadata(q, op.label)
                if op.label:
                    meta["label"] = op.label
                queue_gate_metadata(meta or None)

            elif op.op_type == OpType.Z:
                q = op.qubits[0]
                get_tick_for_qubits([q]).z([q])
                mark_qubits_used([q])
                meta = get_ancilla_gate_metadata(q, op.label)
                if op.label:
                    meta["label"] = op.label
                queue_gate_metadata(meta or None)

            elif op.op_type == OpType.CX:
                qubits = op.qubits
                get_tick_for_qubits(qubits).cx([(qubits[0], qubits[1])])
                mark_qubits_used(qubits)
                meta = get_cx_gate_metadata(qubits[0], qubits[1], op.label)
                if op.label:
                    meta["label"] = op.label
                queue_gate_metadata(meta or None)

            elif op.op_type == OpType.MEASURE:
                q = op.qubits[0]
                get_tick_for_qubits([q]).mz([q])
                mark_qubits_used([q])
                # Label helps identify measurement (e.g., "sx0", "sz0", "final[0]")
                meta = get_ancilla_gate_metadata(q, op.label)
                if op.label:
                    meta["label"] = op.label
                queue_gate_metadata(meta or None)

                # Track measurement index for detectors
                if op.label.startswith("sx"):
                    stab_idx = int(op.label[2:])
                    stab_meas_record[("X", stab_idx, current_round)] = meas_count
                elif op.label.startswith("sz"):
                    stab_idx = int(op.label[2:])
                    stab_meas_record[("Z", stab_idx, current_round)] = meas_count
                elif op.label.startswith("final"):
                    if "final[0]" in op.label:
                        final_meas_start = meas_count
                meas_count += 1

            elif op.op_type == OpType.TICK:
                current_tick_handle = None
                qubits_in_current_tick = set()

        # Apply tick-level and gate-level metadata
        # We use the circuit's set_tick_meta and set_gate_meta methods
        # which modify the ticks in place (unlike get_tick() which returns a copy)
        for tick_idx, tick_meta in all_tick_metadata.items():
            # Set tick-level metadata
            circuit.set_tick_meta(tick_idx, "phase", tick_meta["phase"])
            if tick_meta["round"] >= 0:
                circuit.set_tick_meta(tick_idx, "syndrome_round", tick_meta["round"])
            if tick_meta["cx_round"] > 0:
                circuit.set_tick_meta(tick_idx, "cx_round", tick_meta["cx_round"])

            # Set gate-level metadata (only for gates that have meaningful metadata)
            for gate_idx, gate_meta in enumerate(tick_meta["gates"]):
                if gate_meta:
                    for key, value in gate_meta.items():
                        circuit.set_gate_meta(tick_idx, gate_idx, key, value)

        # Add detector annotations as metadata
        if self.add_detectors:
            geom = patch.geometry
            num_x_anc = len(geom.x_stabilizers)
            deterministic_type_round0 = "Z" if basis.upper() == "Z" else "X"

            detectors = []
            detector_id = 0

            # Syndrome detectors for X stabilizers
            for rnd in range(num_rounds):
                for s in geom.x_stabilizers:
                    curr_idx = stab_meas_record.get(("X", s.index, rnd))
                    if curr_idx is None:
                        continue
                    curr_offset = meas_count - curr_idx

                    if rnd == 0:
                        if deterministic_type_round0 == "X":
                            detectors.append(
                                {
                                    "id": detector_id,
                                    "coords": [s.index, 0, rnd],
                                    "records": [-curr_offset],
                                },
                            )
                            detector_id += 1
                    else:
                        prev_idx = stab_meas_record[("X", s.index, rnd - 1)]
                        prev_offset = meas_count - prev_idx
                        detectors.append(
                            {
                                "id": detector_id,
                                "coords": [s.index, 0, rnd],
                                "records": [-curr_offset, -prev_offset],
                            },
                        )
                        detector_id += 1

            # Syndrome detectors for Z stabilizers
            for rnd in range(num_rounds):
                for s in geom.z_stabilizers:
                    curr_idx = stab_meas_record.get(("Z", s.index, rnd))
                    if curr_idx is None:
                        continue
                    curr_offset = meas_count - curr_idx
                    det_x = num_x_anc + s.index

                    if rnd == 0:
                        if deterministic_type_round0 == "Z":
                            detectors.append(
                                {
                                    "id": detector_id,
                                    "coords": [det_x, 1, rnd],
                                    "records": [-curr_offset],
                                },
                            )
                            detector_id += 1
                    else:
                        prev_idx = stab_meas_record[("Z", s.index, rnd - 1)]
                        prev_offset = meas_count - prev_idx
                        detectors.append(
                            {
                                "id": detector_id,
                                "coords": [det_x, 1, rnd],
                                "records": [-curr_offset, -prev_offset],
                            },
                        )
                        detector_id += 1

            # Final detectors
            if basis.upper() == "Z":
                stabilizers = geom.z_stabilizers
                stab_type = "Z"
                logical_qubits = list(geom.logical_z.data_qubits) if geom.logical_z else []
            else:
                stabilizers = geom.x_stabilizers
                stab_type = "X"
                logical_qubits = list(geom.logical_x.data_qubits) if geom.logical_x else []

            for s in stabilizers:
                data_rec_offsets = [-(meas_count - (final_meas_start + dq)) for dq in s.data_qubits]
                last_syn_idx = stab_meas_record[(stab_type, s.index, num_rounds - 1)]
                syn_offset = -(meas_count - last_syn_idx)
                det_x = s.index if stab_type == "X" else num_x_anc + s.index
                det_y = 0 if stab_type == "X" else 1
                detectors.append(
                    {
                        "id": detector_id,
                        "coords": [det_x, det_y, num_rounds],
                        "records": [*data_rec_offsets, syn_offset],
                    },
                )
                detector_id += 1

            # Logical observable
            logical_rec_offsets = [-(meas_count - (final_meas_start + q)) for q in logical_qubits]
            observables = [
                {
                    "id": 0,
                    "records": logical_rec_offsets,
                },
            ]

            # Store as metadata
            circuit.set_meta("detectors", json.dumps(detectors))
            circuit.set_meta("observables", json.dumps(observables))
            circuit.set_meta("num_measurements", str(meas_count))
            circuit.set_meta("num_detectors", str(len(detectors)))
        circuit.set_meta("basis", basis.upper())
        circuit.set_meta("ancilla_budget", str(allocation.total - len(allocation.data_qubits)))

        return circuit


# Convenience functions


def generate_stim_from_patch(
    patch: SurfacePatch,
    num_rounds: int,
    basis: str = "Z",
    *,
    ancilla_budget: int | None = None,
    p1: float = 0.0,
    p2: float = 0.0,
    p_meas: float = 0.0,
    p_init: float = 0.0,
) -> str:
    """Generate Stim circuit from SurfacePatch.

    Args:
        patch: Surface code patch
        num_rounds: Number of syndrome rounds
        basis: 'Z' or 'X'
        ancilla_budget: Optional cap on simultaneously live ancillas
        p1: Single-qubit error rate
        p2: Two-qubit error rate
        p_meas: Measurement error rate
        p_init: Initialization error rate

    Returns:
        Stim circuit string
    """
    ops, allocation = build_surface_code_circuit(patch, num_rounds, basis, ancilla_budget)
    renderer = StimRenderer(p1=p1, p2=p2, p_meas=p_meas, p_init=p_init)
    return renderer.render(ops, allocation, patch, num_rounds, basis)


def generate_guppy_from_patch(
    patch: SurfacePatch,
    _num_rounds: int = 1,
    _basis: str = "Z",
) -> str:
    """Generate Guppy code from SurfacePatch.

    Generates a full Guppy module with structs, preparation functions,
    syndrome extraction, measurement, logical operators, and factory
    functions (make_memory_z, make_memory_x) for memory experiments.

    Note: num_rounds and basis are accepted for API consistency but not
    used directly. The generated module includes factory functions that
    accept num_rounds as a parameter.

    Args:
        patch: Surface code patch
        _num_rounds: Unused (factory functions accept this at runtime)
        _basis: Unused (module includes both Z and X basis functions)

    Returns:
        Guppy source code string (full module)
    """
    from pecos.guppy.surface import generate_guppy_source

    return generate_guppy_source(patch)


def generate_dag_circuit_from_patch(
    patch: SurfacePatch,
    num_rounds: int,
    basis: str = "Z",
    ancilla_budget: int | None = None,
) -> DagCircuit:
    """Generate PECOS DagCircuit from SurfacePatch.

    Args:
        patch: Surface code patch
        num_rounds: Number of syndrome rounds
        basis: 'Z' or 'X'
        ancilla_budget: Optional cap on simultaneously live ancillas

    Returns:
        PECOS DagCircuit instance
    """
    ops, allocation = build_surface_code_circuit(patch, num_rounds, basis, ancilla_budget)
    renderer = DagCircuitRenderer()
    return renderer.render(ops, allocation, patch, num_rounds, basis)


def generate_tick_circuit_from_patch(
    patch: SurfacePatch,
    num_rounds: int,
    basis: str = "Z",
    *,
    add_detectors: bool = True,
    ancilla_budget: int | None = None,
) -> TickCircuit:
    """Generate PECOS TickCircuit from SurfacePatch.

    TickCircuit has explicit tick boundaries matching Stim's TICK structure.
    This provides a 1:1 correspondence with Stim circuits.

    Detector annotations (similar to Stim's DETECTOR and OBSERVABLE_INCLUDE)
    are stored as circuit metadata:
    - 'detectors': JSON list of {id, coords, records}
    - 'observables': JSON list of {id, records}
    - 'num_measurements': total measurement count
    - 'num_detectors': number of detectors

    Can be converted to DagCircuit via: tick_circuit.to_dag_circuit()
    Metadata is preserved in the DagCircuit.

    Args:
        patch: Surface code patch
        num_rounds: Number of syndrome rounds
        basis: 'Z' or 'X'
        add_detectors: Whether to add detector annotations as metadata
        ancilla_budget: Optional cap on simultaneously live ancillas

    Returns:
        PECOS TickCircuit instance
    """
    ops, allocation = build_surface_code_circuit(patch, num_rounds, basis, ancilla_budget)
    renderer = TickCircuitRenderer(add_detectors=add_detectors)
    return renderer.render(ops, allocation, patch, num_rounds, basis)


def get_detector_descriptors_from_tick_circuit(
    tick_circuit: TickCircuit,
    patch: SurfacePatch,
) -> list[SurfaceDetectorDescriptor]:
    """Return structured detector descriptors for a generated TickCircuit.

    The returned descriptors are cached in TickCircuit metadata when the circuit
    is created by :func:`generate_tick_circuit_from_patch`.

    Example:
        >>> from pecos.qec.surface import SurfacePatch, generate_tick_circuit_from_patch
        >>> patch = SurfacePatch.create(distance=3)
        >>> tc = generate_tick_circuit_from_patch(patch, num_rounds=2, basis="Z")
        >>> len(get_detector_descriptors_from_tick_circuit(tc, patch))
        12
    """
    import json

    cached = tick_circuit.get_meta("detector_descriptors")
    if cached:
        return json.loads(cached)

    detectors = json.loads(tick_circuit.get_meta("detectors") or "[]")
    descriptors = _build_detector_descriptors(detectors, patch)
    tick_circuit.set_meta("detector_descriptors", json.dumps(descriptors))
    return descriptors


def get_observable_descriptors_from_tick_circuit(
    tick_circuit: TickCircuit,
    patch: SurfacePatch,
) -> list[SurfaceObservableDescriptor]:
    """Return structured logical observable descriptors for a TickCircuit.

    Example:
        >>> from pecos.qec.surface import SurfacePatch, generate_tick_circuit_from_patch
        >>> patch = SurfacePatch.create(distance=3)
        >>> tc = generate_tick_circuit_from_patch(patch, num_rounds=2, basis="X")
        >>> get_observable_descriptors_from_tick_circuit(tc, patch)[0]["basis"]
        'X'
    """
    import json

    cached = tick_circuit.get_meta("observable_descriptors")
    if cached:
        return json.loads(cached)

    observables = json.loads(tick_circuit.get_meta("observables") or "[]")
    basis = tick_circuit.get_meta("basis") or "Z"
    descriptors = _build_observable_descriptors(observables, patch, basis)
    tick_circuit.set_meta("observable_descriptors", json.dumps(descriptors))
    return descriptors


def describe_surface_memory_experiment(
    patch: SurfacePatch,
    num_rounds: int,
    basis: str = "Z",
    *,
    add_detectors: bool = True,
    ancilla_budget: int | None = None,
) -> SurfaceMemoryExperimentDescriptor:
    """Return a structured descriptor bundle for a surface-memory experiment.

    This is a convenience wrapper for users who want one public entry point
    that covers patch geometry, stabilizers, logicals, detectors, and
    observables for a generated memory circuit.

    The descriptor helpers are regression-covered on rotated memory circuits and
    also exposed for non-rotated and asymmetric patches created by
    :class:`pecos.qec.surface.SurfacePatch`.

    Example:
        >>> from pecos.qec.surface import SurfacePatch, describe_surface_memory_experiment
        >>> summary = describe_surface_memory_experiment(SurfacePatch.create(distance=3), 2, basis="X")
        >>> summary["basis"]
        'X'
    """
    tick_circuit = generate_tick_circuit_from_patch(
        patch,
        num_rounds=num_rounds,
        basis=basis,
        add_detectors=add_detectors,
        ancilla_budget=ancilla_budget,
    )
    x_stabilizers = list(patch.iter_stabilizer_descriptors("X"))
    z_stabilizers = list(patch.iter_stabilizer_descriptors("Z"))
    logicals = list(patch.iter_logical_descriptors())
    detectors = get_detector_descriptors_from_tick_circuit(tick_circuit, patch)
    observables = get_observable_descriptors_from_tick_circuit(tick_circuit, patch)

    return {
        "patch": patch.get_patch_descriptor(),
        "basis": basis.upper(),
        "num_rounds": num_rounds,
        "ancilla_budget": ancilla_budget,
        "x_stabilizers": x_stabilizers,
        "z_stabilizers": z_stabilizers,
        "stabilizers": x_stabilizers + z_stabilizers,
        "logicals": logicals,
        "detectors": detectors,
        "observables": observables,
    }


def tick_circuit_to_stim(
    tc: TickCircuit,
    *,
    p1: float = 0.0,
    p2: float = 0.0,
    p_meas: float = 0.0,
    p_init: float = 0.0,
) -> str:
    """Convert TickCircuit to Stim circuit string.

    This makes TickCircuit the source of truth for circuit structure,
    with Stim circuit being derived from it for DEM generation.

    Args:
        tc: TickCircuit instance with detector/observable metadata
        p1: Single-qubit error rate
        p2: Two-qubit error rate
        p_meas: Measurement error rate
        p_init: Initialization error rate

    Returns:
        Stim circuit string
    """
    import json
    import math

    lines = []

    simple_gate_map = {
        "H": ("H", "single"),
        "X": ("X", "single"),
        "Y": ("Y", "single"),
        "Z": ("Z", "single"),
        "CX": ("CX", "two"),
        "CY": ("CY", "two"),
        "CZ": ("CZ", "two"),
        "MZ": ("M", "measure"),
        "PZ": ("R", "prep"),
        "QAlloc": ("R", "prep"),
    }

    def _normalized_angle(angle: float) -> float:
        value = angle % math.tau
        if math.isclose(value, math.tau, abs_tol=1e-9):
            return 0.0
        return value

    def _is_close_turn(angle: float, target: float) -> bool:
        return math.isclose(_normalized_angle(angle), target, abs_tol=1e-9)

    def _gate_to_stim(
        gate: object,
    ) -> tuple[list[tuple[str, list[int]]], str | None]:
        gate_name = gate.gate_type.name
        qubits = [int(q) for q in gate.qubits]

        mapped = simple_gate_map.get(gate_name)
        if mapped is not None:
            stim_name, noise_kind = mapped
            return [(stim_name, qubits)], noise_kind

        if gate_name == "RZ":
            if not gate.angles:
                return [], None
            angle = float(gate.angles[0])
            if _is_close_turn(angle, 0.0):
                return [], None
            if _is_close_turn(angle, math.pi):
                return [("Z", qubits)], "single"
            if _is_close_turn(angle, math.pi / 2):
                return [("S", qubits)], "single"
            if _is_close_turn(angle, 3 * math.pi / 2):
                return [("S_DAG", qubits)], "single"
            msg = f"Unsupported traced Clifford RZ angle: {angle!r}"
            raise ValueError(msg)

        if gate_name == "RZZ":
            if not gate.angles:
                return [], None
            angle = float(gate.angles[0])
            if _is_close_turn(angle, 0.0):
                return [], None
            if _is_close_turn(angle, math.pi / 2):
                return [("SQRT_ZZ", qubits)], "two"
            if _is_close_turn(angle, 3 * math.pi / 2):
                return [("SQRT_ZZ_DAG", qubits)], "two"
            msg = f"Unsupported traced Clifford RZZ angle: {angle!r}"
            raise ValueError(msg)

        if gate_name == "R1XY":
            if len(gate.angles) < 2:
                return [], None
            theta = float(gate.angles[0])
            phi = float(gate.angles[1])
            if _is_close_turn(theta, 0.0):
                return [], None
            if _is_close_turn(theta, math.pi):
                if _is_close_turn(phi, 0.0) or _is_close_turn(phi, math.pi):
                    return [("X", qubits)], "single"
                if _is_close_turn(phi, math.pi / 2) or _is_close_turn(phi, 3 * math.pi / 2):
                    return [("Y", qubits)], "single"
            if _is_close_turn(theta, math.pi / 2):
                if _is_close_turn(phi, 0.0):
                    return [("SQRT_X", qubits)], "single"
                if _is_close_turn(phi, math.pi / 2):
                    return [("SQRT_Y", qubits)], "single"
                if _is_close_turn(phi, math.pi):
                    return [("SQRT_X_DAG", qubits)], "single"
                if _is_close_turn(phi, 3 * math.pi / 2):
                    return [("SQRT_Y_DAG", qubits)], "single"
            if _is_close_turn(theta, 3 * math.pi / 2):
                if _is_close_turn(phi, 0.0):
                    return [("SQRT_X_DAG", qubits)], "single"
                if _is_close_turn(phi, math.pi / 2):
                    return [("SQRT_Y_DAG", qubits)], "single"
                if _is_close_turn(phi, math.pi):
                    return [("SQRT_X", qubits)], "single"
                if _is_close_turn(phi, 3 * math.pi / 2):
                    return [("SQRT_Y", qubits)], "single"
            msg = f"Unsupported traced Clifford R1XY angles: theta={theta!r}, phi={phi!r}"
            raise ValueError(msg)

        return [], None

    for tick_idx in range(tc.num_ticks()):
        tick = tc.get_tick(tick_idx)
        for gate in tick.gates():
            instructions, noise_kind = _gate_to_stim(gate)
            if not instructions:
                continue

            qubits = [int(q) for q in gate.qubits]
            qubit_str = " ".join(str(q) for q in qubits)

            if noise_kind == "measure" and p_meas > 0:
                lines.append(f"X_ERROR({p_meas}) {qubit_str}")

            for stim_name, op_qubits in instructions:
                op_qubit_str = " ".join(str(q) for q in op_qubits)
                lines.append(f"{stim_name} {op_qubit_str}")

            if noise_kind == "single" and p1 > 0:
                lines.append(f"DEPOLARIZE1({p1}) {qubit_str}")
            elif noise_kind == "two" and p2 > 0:
                lines.append(f"DEPOLARIZE2({p2}) {qubit_str}")
            elif noise_kind == "prep" and p_init > 0:
                lines.append(f"X_ERROR({p_init}) {qubit_str}")

        # Add TICK after each tick (except the last)
        if tick_idx < tc.num_ticks() - 1:
            lines.append("TICK")

    # Add DETECTOR annotations from TickCircuit metadata
    detectors_json = tc.get_meta("detectors")
    if detectors_json:
        detectors = json.loads(detectors_json)
        for det in detectors:
            coords = det["coords"]
            records = det["records"]
            coord_str = ", ".join(str(c) for c in coords)
            record_str = " ".join(f"rec[{r}]" for r in records)
            lines.append(f"DETECTOR({coord_str}) {record_str}")

    # Add OBSERVABLE_INCLUDE from metadata
    observables_json = tc.get_meta("observables")
    if observables_json:
        observables = json.loads(observables_json)
        for obs in observables:
            obs_id = obs["id"]
            records = obs["records"]
            record_str = " ".join(f"rec[{r}]" for r in records)
            lines.append(f"OBSERVABLE_INCLUDE({obs_id}) {record_str}")

    return "\n".join(lines)


def generate_dem_from_patch(
    patch: SurfacePatch,
    num_rounds: int,
    basis: str = "Z",
    *,
    p: float = 0.01,
) -> str:
    """Generate Detector Error Model from SurfacePatch via Stim.

    This generates a Stim circuit with noise and uses Stim's built-in
    DEM generation for proper circuit-level error analysis.

    Args:
        patch: Surface code patch
        num_rounds: Number of syndrome rounds
        basis: 'Z' or 'X'
        p: Uniform physical error rate

    Returns:
        DEM string in Stim format
    """
    try:
        import stim
    except ImportError as e:
        msg = "Stim is required for DEM generation. Install with: pip install stim"
        raise ImportError(msg) from e

    circuit_str = generate_stim_from_patch(
        patch,
        num_rounds,
        basis,
        p1=p,
        p2=p,
        p_meas=p,
        p_init=p,
    )
    circuit = stim.Circuit(circuit_str)
    return str(circuit.detector_error_model())


def generate_dem_from_tick_circuit_via_pauli_frame(
    tc: TickCircuit,
    *,
    p1: float = 0.01,
    p2: float = 0.01,
    p_meas: float = 0.01,
    p_init: float = 0.01,
) -> str:
    """Generate DEM from TickCircuit using pure Python Pauli frame simulation.

    This is a PECOS-native DEM generator that does not depend on Stim or Rust.
    It uses Pauli frame simulation to track error propagation through
    the circuit and determine which detectors each error triggers.

    The DEM output format matches Stim's DEM format for compatibility
    with PyMatching and other decoders.

    Args:
        tc: TickCircuit with detector/observable metadata
        p1: Single-qubit depolarizing error rate
        p2: Two-qubit depolarizing error rate
        p_meas: Measurement error rate
        p_init: Initialization (prep) error rate

    Returns:
        DEM string in Stim-compatible format
    """
    import json
    from collections import defaultdict

    # Parse detector and observable annotations from metadata
    detectors_json = tc.get_meta("detectors")
    observables_json = tc.get_meta("observables")

    if not detectors_json:
        msg = "TickCircuit must have detector metadata for DEM generation"
        raise ValueError(msg)

    detectors = json.loads(detectors_json)
    observables = json.loads(observables_json) if observables_json else []

    num_measurements = int(tc.get_meta("num_measurements") or "0")

    # Build measurement index -> affected detectors/observables map
    meas_to_detectors: dict[int, list[int]] = defaultdict(list)
    for det in detectors:
        det_id = det["id"]
        for rec in det["records"]:
            abs_meas = num_measurements + rec  # rec is negative
            meas_to_detectors[abs_meas].append(det_id)

    meas_to_observables: dict[int, list[int]] = defaultdict(list)
    for obs in observables:
        obs_id = obs["id"]
        for rec in obs["records"]:
            abs_meas = num_measurements + rec
            meas_to_observables[abs_meas].append(obs_id)

    # Build circuit structure for simulation
    # We need: list of (tick_idx, gate_type, qubits, meas_idx_if_applicable)
    circuit_ops: list[tuple[int, str, list[int], int | None]] = []
    meas_counter = 0

    for tick_idx in range(tc.num_ticks()):
        tick = tc.get_tick(tick_idx)
        for gate in tick.gates():
            gate_name = gate.gate_type.name
            qubits = list(gate.qubits)
            meas_idx = None
            if gate_name == "MZ":
                meas_idx = meas_counter
                meas_counter += 1
            circuit_ops.append((tick_idx, gate_name, qubits, meas_idx))

    def simulate_error(
        start_op_idx: int,
        pauli_frame: dict[int, str],
    ) -> tuple[set[int], set[int]]:
        """Simulate Pauli error propagation from a starting point.

        Args:
            start_op_idx: Index in circuit_ops to start propagation from
            pauli_frame: Initial Pauli frame {qubit: 'X'|'Y'|'Z'}

        Returns:
            (set of triggered detector ids, set of triggered observable ids)
        """
        frame = dict(pauli_frame)  # Copy
        flipped_measurements: set[int] = set()

        for op_idx in range(start_op_idx, len(circuit_ops)):
            _, gate_name, qubits, meas_idx = circuit_ops[op_idx]

            if gate_name in ("QAlloc", "PZ"):
                # Reset clears any error on this qubit
                q = qubits[0]
                frame.pop(q, None)

            elif gate_name == "H":
                # H swaps X ↔ Z, Y → -Y (sign doesn't matter for detection)
                q = qubits[0]
                if q in frame:
                    p = frame[q]
                    if p == "X":
                        frame[q] = "Z"
                    elif p == "Z":
                        frame[q] = "X"
                    # Y stays Y

            elif gate_name == "CX":
                ctrl, targ = qubits[0], qubits[1]
                # CX propagation rules:
                # X_ctrl -> X_ctrl * X_targ
                # Z_targ -> Z_ctrl * Z_targ
                # Y_ctrl = iXZ -> X_ctrl*X_targ * Z_ctrl = Y_ctrl * X_targ
                # Y_targ = iXZ -> X_targ * Z_ctrl*Z_targ = Z_ctrl * Y_targ

                ctrl_p = frame.get(ctrl)
                targ_p = frame.get(targ)

                # Apply CX transformation
                new_ctrl = ctrl_p
                new_targ = targ_p

                if ctrl_p in ("X", "Y"):
                    # X spreads from control to target
                    if targ_p is None:
                        new_targ = "X"
                    elif targ_p == "X":
                        new_targ = None  # X*X = I
                    elif targ_p == "Z":
                        new_targ = "Y"  # X*Z = -iY -> Y
                    elif targ_p == "Y":
                        new_targ = "Z"  # X*Y = iZ -> Z

                if targ_p in ("Z", "Y"):
                    # Z spreads from target to control
                    if ctrl_p is None:
                        new_ctrl = "Z"
                    elif ctrl_p == "Z":
                        new_ctrl = None  # Z*Z = I
                    elif ctrl_p == "X":
                        new_ctrl = "Y"  # Z*X = iY -> Y
                    elif ctrl_p == "Y":
                        new_ctrl = "X"  # Z*Y = -iX -> X

                # Update frame
                if new_ctrl is None:
                    frame.pop(ctrl, None)
                else:
                    frame[ctrl] = new_ctrl
                if new_targ is None:
                    frame.pop(targ, None)
                else:
                    frame[targ] = new_targ

            elif gate_name == "MZ":
                q = qubits[0]
                # Z-basis measurement: X or Y errors flip the result
                if q in frame and frame[q] in ("X", "Y"):
                    flipped_measurements.add(meas_idx)
                # Clear the frame for measured qubit
                frame.pop(q, None)

        # Determine triggered detectors
        triggered_detectors: set[int] = set()
        for meas_idx in flipped_measurements:
            for det_id in meas_to_detectors.get(meas_idx, []):
                # Detector fires if odd number of its measurements are flipped
                if det_id in triggered_detectors:
                    triggered_detectors.remove(det_id)  # Even -> cancel
                else:
                    triggered_detectors.add(det_id)

        triggered_observables: set[int] = set()
        for meas_idx in flipped_measurements:
            for obs_id in meas_to_observables.get(meas_idx, []):
                if obs_id in triggered_observables:
                    triggered_observables.remove(obs_id)
                else:
                    triggered_observables.add(obs_id)

        return triggered_detectors, triggered_observables

    # Collect error mechanisms: (detectors, observables) -> probability
    error_mechanisms: dict[tuple[frozenset[int], frozenset[int]], float] = defaultdict(
        float,
    )

    # Single-qubit Paulis for depolarizing noise
    single_paulis = ["X", "Y", "Z"]
    # Two-qubit Paulis (non-identity on at least one qubit)
    two_paulis = [
        (p1, p2) for p1 in ["I", "X", "Y", "Z"] for p2 in ["I", "X", "Y", "Z"] if not (p1 == "I" and p2 == "I")
    ]

    # Process each gate as a potential error location
    for op_idx, (_tick_idx, gate_name, qubits, meas_idx) in enumerate(circuit_ops):

        if gate_name in ("QAlloc", "PZ") and p_init > 0:
            # Initialization error: X error after prep
            q = qubits[0]
            dets, obs = simulate_error(op_idx + 1, {q: "X"})
            if dets or obs:
                key = (frozenset(dets), frozenset(obs))
                error_mechanisms[key] += p_init

        elif gate_name == "H" and p1 > 0:
            # Single-qubit gate error: depolarizing (each Pauli with prob p1/3)
            q = qubits[0]
            for pauli in single_paulis:
                dets, obs = simulate_error(op_idx + 1, {q: pauli})
                if dets or obs:
                    key = (frozenset(dets), frozenset(obs))
                    error_mechanisms[key] += p1 / 3

        elif gate_name == "CX" and p2 > 0:
            # Two-qubit gate error: depolarizing (each Pauli pair with prob p2/15)
            ctrl, targ = qubits[0], qubits[1]
            for p_ctrl, p_targ in two_paulis:
                frame = {}
                if p_ctrl != "I":
                    frame[ctrl] = p_ctrl
                if p_targ != "I":
                    frame[targ] = p_targ
                dets, obs = simulate_error(op_idx + 1, frame)
                if dets or obs:
                    key = (frozenset(dets), frozenset(obs))
                    error_mechanisms[key] += p2 / 15

        elif gate_name == "MZ" and p_meas > 0:
            # Measurement error: bit flip (affects this measurement directly)
            # This is before the measurement is taken, so we track it as X error
            # that is immediately measured
            q = qubits[0]
            # For measurement error, we directly flip this measurement
            dets = set()
            obs = set()
            for det_id in meas_to_detectors.get(meas_idx, []):
                dets.add(det_id)
            for obs_id in meas_to_observables.get(meas_idx, []):
                obs.add(obs_id)
            if dets or obs:
                key = (frozenset(dets), frozenset(obs))
                error_mechanisms[key] += p_meas

    # Generate DEM output
    lines = []

    # Add detector coordinate annotations
    for det in detectors:
        coords = det["coords"]
        coord_str = ", ".join(str(c) for c in coords)
        lines.append(f"detector({coord_str}) D{det['id']}")

    # Add logical observable
    lines.extend(f"logical_observable L{obs['id']}" for obs in observables)

    # Add error mechanisms (combine same-effect errors)
    for (dets, obs), prob in sorted(
        error_mechanisms.items(),
        key=lambda x: (sorted(x[0][0]), sorted(x[0][1])),
    ):
        if prob > 0 and (dets or obs):
            det_str = " ".join(f"D{d}" for d in sorted(dets))
            obs_str = " ".join(f"L{o}" for o in sorted(obs))
            targets = f"{det_str} {obs_str}".strip()
            lines.append(f"error({prob:.6g}) {targets}")

    return "\n".join(lines)


def generate_dem_from_tick_circuit_via_stim(
    tc: TickCircuit,
    *,
    p1: float = 0.01,
    p2: float = 0.01,
    p_meas: float = 0.01,
    p_init: float = 0.01,
    decompose_errors: bool = True,
    maximal_decomposition: bool = False,
) -> str:
    """Generate DEM from TickCircuit via Stim conversion.

    This uses TickCircuit as the source of truth for circuit structure,
    converts to Stim format, and uses Stim's DEM generator for full
    circuit-level noise analysis.

    Args:
        tc: TickCircuit with detector/observable metadata
        p1: Single-qubit depolarizing error rate
        p2: Two-qubit depolarizing error rate
        p_meas: Measurement error rate
        p_init: Initialization (prep) error rate
        decompose_errors: If True (default), ask Stim to decompose hyperedge
            errors into graphlike components. Set to False to preserve raw
            hyperedges.
        maximal_decomposition: If True, post-process Stim's graphlike output
            into the same singleton-preferring maximal decomposition used by
            the native DEM path. Ignored when False.

    Returns:
        DEM string in Stim format
    """
    try:
        import stim
    except ImportError as e:
        msg = "Stim is required for this function. Install with: pip install stim"
        raise ImportError(msg) from e

    stim_str = tick_circuit_to_stim(tc, p1=p1, p2=p2, p_meas=p_meas, p_init=p_init)
    circuit = stim.Circuit(stim_str)
    dem = circuit.detector_error_model(decompose_errors=decompose_errors or maximal_decomposition)
    if maximal_decomposition:
        return _maximally_decompose_graphlike_dem(str(dem))
    return str(dem)


def _extract_measurement_order(tc: TickCircuit) -> list[int]:
    """Extract the measurement order from a TickCircuit.

    Returns a list of qubit indices in the order they were measured.
    measurement_order[i] is the qubit measured at TickCircuit measurement index i.

    This allows proper mapping between record offsets (which use TickCircuit
    measurement order) and influence map indices (which use DAG topological order).

    Args:
        tc: TickCircuit to extract measurement order from.

    Returns:
        List of qubit indices in measurement execution order.
    """
    measurement_order = []

    for tick_idx in range(tc.num_ticks()):
        tick = tc.get_tick(tick_idx)
        if tick is None:
            continue
        gates = tick.gates()
        for gate in gates:
            gate_type = str(gate.gate_type)
            if "MZ" in gate_type:
                # Add each measured qubit to the order
                for qubit in gate.qubits:
                    # Qubit might be an int or a QubitId object
                    if hasattr(qubit, "index"):
                        measurement_order.append(qubit.index())
                    else:
                        measurement_order.append(int(qubit))

    return measurement_order


def get_measurement_order_from_tick_circuit(tc: TickCircuit) -> list[int]:
    """Public wrapper returning the TickCircuit measurement execution order."""
    return _extract_measurement_order(tc)


def _maximally_decompose_graphlike_dem(dem_text: str) -> str:
    """Prefer singleton graphlike components when the decomposed DEM exposes them.

    This is a formatting-level refinement over the standard decomposed DEM:
    when a 2-detector direct mechanism `D_i D_j` has corresponding singleton
    components already present in the DEM, prefer `D_i ^ D_j` (or the boundary
    form `D_i L0 ^ D_j L0`) instead.
    """
    standalone_detectors: set[str] = set()
    det_l0_detectors: set[str] = set()
    lines = dem_text.splitlines()

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("error("):
            continue
        payload = stripped.split(")", 1)[1].strip()
        if "^" in payload:
            continue
        tokens = payload.split()
        detectors = [token for token in tokens if token.startswith("D")]
        logicals = [token for token in tokens if token.startswith("L")]
        if len(detectors) == 1 and not logicals:
            standalone_detectors.add(detectors[0])
        elif len(detectors) == 1 and logicals == ["L0"]:
            det_l0_detectors.add(detectors[0])

    rewritten_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("error("):
            rewritten_lines.append(line)
            continue
        prefix, payload = stripped.split(")", 1)
        payload = payload.strip()
        if "^" in payload:
            rewritten_lines.append(line)
            continue
        tokens = payload.split()
        detectors = [token for token in tokens if token.startswith("D")]
        logicals = [token for token in tokens if token.startswith("L")]
        if len(detectors) == 2 and not logicals:
            d0, d1 = detectors
            replacement: str | None = None
            if d0 in standalone_detectors and d1 in standalone_detectors:
                replacement = f"{d0} ^ {d1}"
            elif d0 in det_l0_detectors and d1 in det_l0_detectors:
                replacement = f"{d0} L0 ^ {d1} L0"
            if replacement is not None:
                rewritten_lines.append(f"{prefix}) {replacement}")
                continue
        rewritten_lines.append(line)

    return "\n".join(rewritten_lines)


def generate_dem_from_tick_circuit(
    tc: TickCircuit,
    *,
    p1: float = 0.01,
    p2: float = 0.01,
    p_meas: float = 0.01,
    p_init: float = 0.01,
    decompose_errors: bool = True,
    maximal_decomposition: bool = False,
) -> str:
    """Generate DEM from TickCircuit using pre-defined detector annotations.

    This is the main PECOS-native DEM generator. It uses the Rust
    DemBuilder for efficient analysis, which handles per-qubit fault
    locations and maps fault effects to the pre-defined detector annotations
    in the TickCircuit metadata.

    When decompose_errors=True (default), hyperedge errors (affecting 3+
    detectors) are decomposed into graphlike errors (1-2 detectors) using
    the `^` separator syntax. This is necessary for MWPM decoders which
    only work on graphs, not hypergraphs.

    When maximal_decomposition=True, ALL mechanisms (including 2-detector)
    are decomposed into single-detector components when possible. This uses
    only single-detector components that exist as standalone entries in the
    DEM. For boundary detectors where the only available component is
    `D_i L0`, the L0 terms naturally XOR away when combined.

    Args:
        tc: TickCircuit with detector/observable metadata (required)
        p1: Single-qubit depolarizing error rate
        p2: Two-qubit depolarizing error rate
        p_meas: Measurement error rate
        p_init: Initialization (prep) error rate
        decompose_errors: If True (default), decompose hyperedge errors into
            graphlike components using the `^` separator. Set to False to
            output raw hyperedges. Ignored if maximal_decomposition=True.
        maximal_decomposition: If True, maximally decompose all mechanisms
            into single-detector components. This produces output similar
            to other tools that prefer maximal decomposition.

    Returns:
        DEM string in Stim-compatible format
    """
    from pecos.qec import DagFaultAnalyzer, DemBuilder

    # Get detector and observable metadata
    detectors_json = tc.get_meta("detectors")
    observables_json = tc.get_meta("observables")

    if not detectors_json:
        msg = "TickCircuit must have detector metadata for DEM generation"
        raise ValueError(msg)

    num_measurements = int(tc.get_meta("num_measurements") or "0")

    # Extract measurement order from TickCircuit: list of qubits in measurement execution order
    # This allows proper mapping between record offsets (TickCircuit order) and
    # influence map indices (DAG topological order).
    measurement_order = _extract_measurement_order(tc)

    # Convert TickCircuit to DagCircuit and build influence map
    dag = tc.to_dag_circuit()
    analyzer = DagFaultAnalyzer(dag)
    influence_map = analyzer.build_influence_map()

    # Build DEM using Rust DemBuilder
    builder = DemBuilder(influence_map)
    builder.with_noise(p1, p2, p_meas, p_init)
    builder.with_num_measurements(num_measurements)
    builder.with_measurement_order(measurement_order)
    builder.with_detectors_json(detectors_json)
    if observables_json:
        builder.with_observables_json(observables_json)

    dem = builder.build_with_source_tracking()

    if maximal_decomposition:
        return _maximally_decompose_graphlike_dem(dem.to_string_decomposed())
    if decompose_errors:
        return dem.to_string_decomposed()
    return dem.to_string()


def generate_dem_from_tick_circuit_via_autodetection(
    tc: TickCircuit,
    *,
    logical_z_qubits: list[int] | None = None,
    logical_x_qubits: list[int] | None = None,
    p1: float = 0.01,
    p2: float = 0.01,
    p_meas: float = 0.01,
    p_init: float = 0.01,
) -> str:
    """Generate DEM from TickCircuit using auto-discovered detectors.

    This uses the Rust InfluenceBuilder which performs symbolic simulation
    to automatically discover deterministic measurements and define detectors
    from them. This is useful when detector annotations are not available.

    Unlike generate_dem_from_tick_circuit which uses pre-defined detector
    annotations, this function discovers detectors automatically. The resulting
    DEM may have a different detector structure than Stim-generated DEMs.

    Args:
        tc: TickCircuit (detector annotations not required)
        logical_z_qubits: Qubit indices for logical Z operator (for X error tracking)
        logical_x_qubits: Qubit indices for logical X operator (for Z error tracking)
        p1: Single-qubit depolarizing error rate
        p2: Two-qubit depolarizing error rate
        p_meas: Measurement error rate
        p_init: Initialization (prep) error rate

    Returns:
        DEM string in Stim-compatible format
    """
    from collections import defaultdict

    from pecos.qec import PAULI_X, PAULI_Y, PAULI_Z, InfluenceBuilder

    # Convert TickCircuit to DagCircuit
    dag = tc.to_dag_circuit()

    # Build influence map with auto-discovered detectors
    builder = InfluenceBuilder(dag)
    if logical_z_qubits:
        builder.with_logical_z(logical_z_qubits)
    if logical_x_qubits:
        builder.with_logical_x(logical_x_qubits)
    influence_map = builder.build()

    # Get all fault locations and auto-discovered detectors
    locations = influence_map.get_locations()
    num_detectors = influence_map.num_detectors
    num_logicals = influence_map.num_logicals

    # Collect error mechanisms: (detectors, logicals) -> probability
    error_mechanisms: dict[tuple[frozenset[int], frozenset[int]], float] = defaultdict(
        float,
    )

    # Process each fault location
    for loc_idx, loc in enumerate(locations):
        gate_type = loc.gate_type

        if "PZ" in gate_type or "QAlloc" in gate_type:
            if p_init <= 0:
                continue
            for pauli in [PAULI_X]:
                dets = set(influence_map.get_detector_indices(loc_idx, pauli))
                logs = set(influence_map.get_logical_indices(loc_idx, pauli))
                if dets or logs:
                    key = (frozenset(dets), frozenset(logs))
                    error_mechanisms[key] += p_init

        elif "MZ" in gate_type:
            if p_meas <= 0:
                continue
            for pauli in [PAULI_X]:
                dets = set(influence_map.get_detector_indices(loc_idx, pauli))
                logs = set(influence_map.get_logical_indices(loc_idx, pauli))
                if dets or logs:
                    key = (frozenset(dets), frozenset(logs))
                    error_mechanisms[key] += p_meas

        elif "CX" in gate_type:
            if p2 <= 0:
                continue
            for pauli in [PAULI_X, PAULI_Y, PAULI_Z]:
                dets = set(influence_map.get_detector_indices(loc_idx, pauli))
                logs = set(influence_map.get_logical_indices(loc_idx, pauli))
                if dets or logs:
                    key = (frozenset(dets), frozenset(logs))
                    error_mechanisms[key] += p2 / 3

        elif "H" in gate_type:
            if p1 <= 0:
                continue
            for pauli in [PAULI_X, PAULI_Y, PAULI_Z]:
                dets = set(influence_map.get_detector_indices(loc_idx, pauli))
                logs = set(influence_map.get_logical_indices(loc_idx, pauli))
                if dets or logs:
                    key = (frozenset(dets), frozenset(logs))
                    error_mechanisms[key] += p1 / 3

    # Generate DEM output
    # Add detector declarations (auto-discovered, no coordinates)
    lines = [f"detector D{det_idx}" for det_idx in range(num_detectors)]

    # Add logical observables
    lines.extend(f"logical_observable L{log_idx}" for log_idx in range(num_logicals))

    # Add error mechanisms
    for (dets, logs), prob in sorted(
        error_mechanisms.items(),
        key=lambda x: (sorted(x[0][0]), sorted(x[0][1])),
    ):
        if prob > 0 and (dets or logs):
            det_str = " ".join(f"D{d}" for d in sorted(dets))
            log_str = " ".join(f"L{log_idx}" for log_idx in sorted(logs))
            targets = f"{det_str} {log_str}".strip()
            lines.append(f"error({prob:.6g}) {targets}")

    return "\n".join(lines)
