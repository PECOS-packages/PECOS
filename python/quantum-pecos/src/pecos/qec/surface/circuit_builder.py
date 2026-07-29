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
    from collections.abc import Mapping

# `_batched_stabilizers` and `_normalize_ancilla_budget` are imported from
# the shared `_ancilla_batching` helper so this builder and the Guppy
# emitter (`pecos.guppy.surface`) compute identical batches by
# construction. The local aliases preserve existing call sites; do not
# fork the partitioning logic.
from pecos.qec.surface._ancilla_batching import (
    batched_stabilizers as _batched_stabilizers,
)
from pecos.qec.surface._ancilla_batching import (
    normalize_ancilla_budget as _normalize_ancilla_budget,
)
from pecos.qec.surface._check_plan import (
    ancilla_schedule_for_check_plan,
    cnot_round_order_for_check_plan,
    require_current_surface_check_plan_renderer,
    resolve_surface_check_plan,
)
from pecos.qec.surface._clifford_deformation import (
    resolve_surface_clifford_frame,
)

# Stabilizer geometry helpers live in the low-level patch module (single
# source of truth). Only the two used by the circuit renderer are imported
# here; the full set is exported publicly from the package __init__.
from pecos.qec.surface.patch import (
    get_stabilizer_region,
    get_stabilizer_touch_label,
)
from pecos.quantum import PHYSICAL_DURATION_META_KEY

if TYPE_CHECKING:
    from pecos.qec.surface._check_plan import ResolvedSurfaceCheckPlan
    from pecos.qec.surface._clifford_deformation import (
        PauliAxis,
        ResolvedPauliCheck,
        ResolvedSurfaceCliffordFrame,
    )
    from pecos.qec.surface._twirl_config import TwirlConfig
    from pecos.qec.surface.patch import (
        LogicalDescriptor,
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
    F = auto()  # face Clifford
    FDG = auto()  # face Clifford dagger
    SX = auto()  # sqrt X
    SXDG = auto()  # sqrt X dagger
    SY = auto()  # sqrt Y
    SYDG = auto()  # sqrt Y dagger
    SZ = auto()  # sqrt Z / phase
    SZDG = auto()  # sqrt Z dagger
    X = auto()  # Pauli X
    Z = auto()  # Pauli Z

    # Two-qubit gates
    CX = auto()  # CNOT
    SZZ = auto()  # sqrt ZZ
    SZZDG = auto()  # sqrt ZZ dagger

    # Measurement
    MEASURE = auto()  # Destructive measurement

    # Structural
    TICK = auto()  # Layer separator
    COMMENT = auto()  # Comment/annotation

    # Annotation: declares a candidate tracked Pauli at this circuit position.
    # The propagator records its forward propagation to detectors and
    # observables; per-shot "did this Pauli fire?" is consumed at sampling
    # time. ``qubits`` is the single data qubit the Pauli acts on; ``label``
    # carries the Pauli kind and site as ``"{X|Y|Z}@s<site_idx>"``.
    TRACKED_PAULI = auto()


@dataclass
class SurfaceCircuitStep:
    """A surface-code circuit builder step."""

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


@dataclass(frozen=True)
class SzzForwardFlowPulse:
    """One pending-Clifford discharge site in the SZZ device-flow model."""

    host_index: int
    host_op_type: str
    host_label: str
    qubit: int
    kind: str
    pending_clifford: str


@dataclass(frozen=True)
class SzzForwardFlowSummary:
    """Pulse accounting for the SZZ pending-Clifford forward-flow model."""

    abstract_single_qubit_ops: int
    physical_prefix_pulses: int
    two_qubit_prefix_pulses: int
    measurement_prefix_pulses: int
    virtual_z_two_qubit_carries: int
    virtual_z_measure_discards: int
    two_qubit_gates: int
    measurements: int
    prep_events: int
    free_standing_single_qubit_ops: int
    pulses: tuple[SzzForwardFlowPulse, ...]


@dataclass(frozen=True, order=True)
class SzzTouchSign:
    """Signed SZZ touch in the v1 surface-memory sign convention."""

    stabilizer_type: str
    stabilizer_index: int
    data_qubit: int
    sign: int


@dataclass(frozen=True, order=True)
class SzzBoundaryCompensation:
    """Analysis-only compensation for an odd uncompensated boundary residual."""

    stabilizer_type: str
    stabilizer_index: int
    data_qubit: int
    gate: str


@dataclass(frozen=True, order=True)
class SzzClass2Residual:
    """Analysis-only class-2 data residual of the uncompensated sign vector."""

    stabilizer_type: str
    data_qubit: int
    pauli: str


@dataclass(frozen=True)
class SzzResidualPlan:
    """Validated SZZ sign convention and uncompensated residual bookkeeping.

    The active SZZ template cancels data residuals with immediate per-touch
    compensation; no class-2 residual stream is emitted by the circuit.
    """

    signs: tuple[SzzTouchSign, ...]
    boundary_compensations: tuple[SzzBoundaryCompensation, ...]
    class2_residuals: tuple[SzzClass2Residual, ...]


def _normalize_interaction_basis(interaction_basis: str) -> str:
    """Validate and normalize the surface two-qubit interaction basis."""
    normalized = interaction_basis.lower()
    if normalized not in {"cx", "szz"}:
        msg = f"interaction_basis must be 'cx' or 'szz', got {interaction_basis!r}"
        raise ValueError(msg)
    return normalized


def _szz_residual_class(sum_signs: int) -> str:
    """Classify a signed residual sum modulo four."""
    residue = sum_signs % 4
    if residue == 0:
        return "identity"
    if residue == 2:
        return "pauli"
    return "odd"


def _iter_surface_stabilizer_touches(patch: SurfacePatch) -> list[tuple[str, int, tuple[int, ...], bool]]:
    """Return stabilizer touch rows in deterministic X-then-Z order."""
    geom = patch.geometry
    rows: list[tuple[str, int, tuple[int, ...], bool]] = []
    rows.extend(("X", stab.index, tuple(stab.data_qubits), bool(stab.is_boundary)) for stab in geom.x_stabilizers)
    rows.extend(("Z", stab.index, tuple(stab.data_qubits), bool(stab.is_boundary)) for stab in geom.z_stabilizers)
    return rows


def _default_szz_sign_vector(patch: SurfacePatch) -> tuple[SzzTouchSign, ...]:
    """Return the v1 hard-coded SZZ sign vector.

    Bulk checks use all ``SZZ``. Boundary checks use one ``SZZdg`` on the
    second data operand, giving ancilla class 0 while preserving a stable,
    geometry-derived convention.
    """
    signs: list[SzzTouchSign] = []
    for stabilizer_type, stabilizer_index, data_qubits, is_boundary in _iter_surface_stabilizer_touches(patch):
        if is_boundary and len(data_qubits) != 2:
            msg = (
                "SZZ v1 expects boundary stabilizers to have weight 2; "
                f"{stabilizer_type}{stabilizer_index} has weight {len(data_qubits)}"
            )
            raise ValueError(msg)
        for touch_index, data_qubit in enumerate(data_qubits):
            sign = -1 if is_boundary and touch_index == 1 else 1
            signs.append(
                SzzTouchSign(
                    stabilizer_type=stabilizer_type,
                    stabilizer_index=stabilizer_index,
                    data_qubit=data_qubit,
                    sign=sign,
                ),
            )
    return tuple(sorted(signs))


def _boundary_first_szz_sign_vector(patch: SurfacePatch) -> tuple[SzzTouchSign, ...]:
    """Return the SZZ sign vector with boundary daggers on first operands.

    This is the first non-default SZZ/SZZdg source-level check plan. It keeps
    the same schedule and compensation model as the default plan but changes
    the concrete signed SZZ touch chosen on each weight-2 boundary check.
    """
    signs: list[SzzTouchSign] = []
    for stabilizer_type, stabilizer_index, data_qubits, is_boundary in _iter_surface_stabilizer_touches(patch):
        if is_boundary and len(data_qubits) != 2:
            msg = (
                "SZZ boundary-first expects boundary stabilizers to have weight 2; "
                f"{stabilizer_type}{stabilizer_index} has weight {len(data_qubits)}"
            )
            raise ValueError(msg)
        for touch_index, data_qubit in enumerate(data_qubits):
            sign = -1 if is_boundary and touch_index == 0 else 1
            signs.append(
                SzzTouchSign(
                    stabilizer_type=stabilizer_type,
                    stabilizer_index=stabilizer_index,
                    data_qubit=data_qubit,
                    sign=sign,
                ),
            )
    return tuple(sorted(signs))


def _validate_szz_sign_vector(
    patch: SurfacePatch,
    signs: tuple[SzzTouchSign, ...],
) -> SzzResidualPlan:
    """Validate an SZZ sign vector and derive fixed residual bookkeeping."""
    expected_keys = {
        (stabilizer_type, stabilizer_index, data_qubit)
        for stabilizer_type, stabilizer_index, data_qubits, _is_boundary in _iter_surface_stabilizer_touches(patch)
        for data_qubit in data_qubits
    }
    seen_keys: set[tuple[str, int, int]] = set()
    check_sums: dict[tuple[str, int], int] = {}
    data_sums: dict[tuple[str, int], int] = {}
    data_touch_counts: dict[tuple[str, int], int] = {}

    for entry in signs:
        if entry.sign not in {-1, 1}:
            msg = f"SZZ sign entries must be +/-1, got {entry.sign!r} for {entry}"
            raise ValueError(msg)
        key = (entry.stabilizer_type, entry.stabilizer_index, entry.data_qubit)
        if key in seen_keys:
            msg = f"duplicate SZZ sign entry for touch {key}"
            raise ValueError(msg)
        seen_keys.add(key)
        check_key = (entry.stabilizer_type, entry.stabilizer_index)
        data_key = (entry.stabilizer_type, entry.data_qubit)
        check_sums[check_key] = check_sums.get(check_key, 0) + entry.sign
        data_sums[data_key] = data_sums.get(data_key, 0) + entry.sign
        data_touch_counts[data_key] = data_touch_counts.get(data_key, 0) + 1

    missing = sorted(expected_keys - seen_keys)
    extra = sorted(seen_keys - expected_keys)
    if missing or extra:
        msg = f"SZZ sign vector must cover exactly the surface touches; missing={missing}, extra={extra}"
        raise ValueError(msg)

    for (stabilizer_type, stabilizer_index), sum_signs in sorted(check_sums.items()):
        residual_class = _szz_residual_class(sum_signs)
        if residual_class != "identity":
            msg = (
                "SZZ sign vector rejected: "
                f"{stabilizer_type}{stabilizer_index} ancilla residual is {residual_class} "
                f"(sum={sum_signs}, mod4={sum_signs % 4}); v1 has no record-flip checks"
            )
            raise ValueError(msg)

    compensations: list[SzzBoundaryCompensation] = []
    class2_residuals: list[SzzClass2Residual] = []
    for (stabilizer_type, data_qubit), sum_signs in sorted(data_sums.items()):
        residual_class = _szz_residual_class(sum_signs)
        if residual_class == "identity":
            continue
        if residual_class == "pauli":
            class2_residuals.append(
                SzzClass2Residual(
                    stabilizer_type=stabilizer_type,
                    data_qubit=data_qubit,
                    pauli="X" if stabilizer_type == "X" else "Z",
                ),
            )
            continue
        if data_touch_counts[(stabilizer_type, data_qubit)] != 1:
            msg = (
                "SZZ sign vector rejected: odd residual on a non-boundary data class "
                f"{stabilizer_type}, data={data_qubit}, sum={sum_signs}"
            )
            raise ValueError(msg)
        touch = next(
            entry for entry in signs if entry.stabilizer_type == stabilizer_type and entry.data_qubit == data_qubit
        )
        gate = {
            ("X", 1): "SXDG",
            ("X", -1): "SX",
            ("Z", 1): "SZDG",
            ("Z", -1): "SZ",
        }[(stabilizer_type, touch.sign)]
        compensations.append(
            SzzBoundaryCompensation(
                stabilizer_type=touch.stabilizer_type,
                stabilizer_index=touch.stabilizer_index,
                data_qubit=touch.data_qubit,
                gate=gate,
            ),
        )

    return SzzResidualPlan(
        signs=tuple(sorted(signs)),
        boundary_compensations=tuple(sorted(compensations)),
        class2_residuals=tuple(sorted(class2_residuals)),
    )


def _default_szz_residual_plan(patch: SurfacePatch) -> SzzResidualPlan:
    """Return the validated v1 SZZ residual plan for a patch."""
    return _validate_szz_sign_vector(patch, _default_szz_sign_vector(patch))


def _szz_residual_plan_for_check_plan(
    patch: SurfacePatch,
    resolved_plan: ResolvedSurfaceCheckPlan,
) -> SzzResidualPlan:
    """Return the concrete SZZ residual plan for a resolved check plan."""
    if resolved_plan.interaction_basis != "szz":
        msg = f"SZZ residual plans require interaction_basis='szz', got {resolved_plan.interaction_basis!r}"
        raise ValueError(msg)

    pattern = str(resolved_plan.synthesis_identity["szz_phase_pattern"])
    if pattern == "standard":
        return _default_szz_residual_plan(patch)
    if pattern == "boundary-first":
        return _validate_szz_sign_vector(patch, _boundary_first_szz_sign_vector(patch))

    msg = f"unsupported SZZ phase pattern {pattern!r} for check_plan={resolved_plan.plan_id!r}"
    raise NotImplementedError(msg)


def _resolve_szz_clifford_frame_for_builder(
    patch: SurfacePatch,
    *,
    interaction_basis: str,
    clifford_frame_policy: str | None,
) -> ResolvedSurfaceCliffordFrame | None:
    """Resolve an optional source-level Clifford frame for SZZ rendering."""
    if clifford_frame_policy is None:
        return None
    if interaction_basis != "szz":
        msg = "clifford_frame_policy currently requires interaction_basis='szz'"
        raise NotImplementedError(msg)
    return resolve_surface_clifford_frame(patch, policy=clifford_frame_policy)


def _szz_memory_physical_axis(
    basis: str,
    resolved_clifford_frame: ResolvedSurfaceCliffordFrame | None,
) -> PauliAxis:
    """Return the uniform physical axis for a source memory basis, if any."""
    source_basis = basis.upper()
    if source_basis not in {"X", "Z"}:
        msg = f"basis must be 'X' or 'Z', got {basis!r}"
        raise ValueError(msg)
    if resolved_clifford_frame is None:
        return source_basis  # type: ignore[return-value]

    axes = {frame.image(source_basis).axis for frame in resolved_clifford_frame.data_frames}
    if len(axes) != 1:
        msg = (
            f"clifford frame policy {resolved_clifford_frame.policy!r} maps "
            f"source {source_basis}-memory to mixed data measurement axes "
            f"{sorted(axes)}; call _szz_memory_physical_axis_for_data instead"
        )
        raise NotImplementedError(msg)
    return next(iter(axes))


def _szz_memory_physical_axis_for_data(
    basis: str,
    resolved_clifford_frame: ResolvedSurfaceCliffordFrame | None,
    data_idx: int,
) -> PauliAxis:
    """Return the physical prep/readout axis for one source-basis data qubit."""
    source_basis = basis.upper()
    if source_basis not in {"X", "Z"}:
        msg = f"basis must be 'X' or 'Z', got {basis!r}"
        raise ValueError(msg)
    if resolved_clifford_frame is None:
        return source_basis  # type: ignore[return-value]
    try:
        frame = resolved_clifford_frame.data_frames[data_idx]
    except IndexError as exc:
        msg = (
            f"data qubit {data_idx} is outside resolved frame with "
            f"{len(resolved_clifford_frame.data_frames)} data frames"
        )
        raise ValueError(msg) from exc
    return frame.image(source_basis).axis


def _propagate_szz_frame_bits(x_a: bool, z_a: bool, x_b: bool, z_b: bool) -> tuple[bool, bool, bool, bool]:
    """Propagate local Pauli-frame bits through uncompensated SZZ/SZZdg."""
    common = x_a ^ x_b
    return x_a, z_a ^ common, x_b, z_b ^ common


def _propagate_compensated_szz_frame_bits(x_a: bool, z_a: bool, x_b: bool, z_b: bool) -> tuple[bool, bool, bool, bool]:
    """Propagate local Pauli-frame bits through the compensated CZ-equivalent interaction."""
    return x_a, z_a ^ x_b, x_b, z_b ^ x_a


def _propagate_sxx_frame_bits(x_a: bool, z_a: bool, x_b: bool, z_b: bool) -> tuple[bool, bool, bool, bool]:
    """Propagate local Pauli-frame bits through the SXX/SXXdg mirror."""
    common = z_a ^ z_b
    return x_a ^ common, z_a, x_b ^ common, z_b


_SIGNED_PAULI_NAMES = {
    1: "X",
    -1: "-X",
    2: "Y",
    -2: "-Y",
    3: "Z",
    -3: "-Z",
}
_SZZ_FLOW_IDENTITY: tuple[int, int] = (1, 3)
_SZZ_FLOW_SINGLE_QUBIT_GATES = {
    OpType.H,
    OpType.SX,
    OpType.SXDG,
    OpType.SZ,
    OpType.SZDG,
    OpType.X,
    OpType.Z,
}
_SZZ_FLOW_GATE_ACTIONS: dict[OpType, dict[int, int]] = {
    OpType.H: {1: 3, 2: -2, 3: 1},
    OpType.F: {1: 2, 2: 3, 3: 1},
    OpType.FDG: {1: 3, 2: 1, 3: 2},
    OpType.SX: {1: 1, 2: 3, 3: -2},
    OpType.SXDG: {1: 1, 2: -3, 3: 2},
    OpType.SY: {1: -3, 2: 2, 3: 1},
    OpType.SYDG: {1: 3, 2: 2, 3: -1},
    OpType.SZ: {1: 2, 2: -1, 3: 3},
    OpType.SZDG: {1: -2, 2: 1, 3: 3},
    OpType.X: {1: 1, 2: -2, 3: -3},
    OpType.Z: {1: -1, 2: -2, 3: 3},
}


def _szz_flow_apply_gate_to_pauli(op_type: OpType, pauli: int) -> int:
    sign = -1 if pauli < 0 else 1
    return sign * _SZZ_FLOW_GATE_ACTIONS[op_type][abs(pauli)]


def _szz_flow_compose_pending_gate(pending: tuple[int, int], op_type: OpType) -> tuple[int, int]:
    """Append one 1q Clifford to a pending signed-Pauli-image register."""
    return (
        _szz_flow_apply_gate_to_pauli(op_type, pending[0]),
        _szz_flow_apply_gate_to_pauli(op_type, pending[1]),
    )


def _szz_flow_is_virtual_z(pending: tuple[int, int]) -> bool:
    """Return whether a pending Clifford is a zero-pulse virtual-Z update."""
    return pending[1] == 3


def _szz_flow_clifford_name(pending: tuple[int, int]) -> str:
    return f"X->{_SIGNED_PAULI_NAMES[pending[0]]},Z->{_SIGNED_PAULI_NAMES[pending[1]]}"


def _analyze_szz_forward_flow(ops: list[SurfaceCircuitStep]) -> SzzForwardFlowSummary:
    """Analyze SZZ pending-Clifford forward-flow pulse accounting.

    The abstract SZZ template intentionally contains free-standing 1q
    Cliffords. The SZZ device model executes those Cliffords only as prefixes
    of the next SZZ/SZZdg or MZ on that qubit. Virtual-Z pending Cliffords carry
    through SZZ/SZZdg and are discarded at MZ.
    """
    pending_by_qubit: dict[int, tuple[int, int]] = {}
    pulses: list[SzzForwardFlowPulse] = []
    abstract_single_qubit_ops = 0
    physical_prefix_pulses = 0
    two_qubit_prefix_pulses = 0
    measurement_prefix_pulses = 0
    virtual_z_two_qubit_carries = 0
    virtual_z_measure_discards = 0
    two_qubit_gates = 0
    measurements = 0
    prep_events = 0

    def pending_for(q: int) -> tuple[int, int]:
        return pending_by_qubit.setdefault(q, _SZZ_FLOW_IDENTITY)

    def pulse_event(
        *,
        host_index: int,
        op: SurfaceCircuitStep,
        qubit: int,
        kind: str,
        pending: tuple[int, int],
    ) -> SzzForwardFlowPulse:
        return SzzForwardFlowPulse(
            host_index=host_index,
            host_op_type=op.op_type.name,
            host_label=op.label,
            qubit=qubit,
            kind=kind,
            pending_clifford=_szz_flow_clifford_name(pending),
        )

    def reset_for_prep(q: int, op: SurfaceCircuitStep) -> None:
        current = pending_for(q)
        if current != _SZZ_FLOW_IDENTITY:
            msg = (
                "SZZ forward-flow cannot reset a qubit with a pending "
                f"Clifford: q={q}, pending={_szz_flow_clifford_name(current)}, "
                f"op={op.op_type.name} {op.label!r}"
            )
            raise ValueError(msg)
        pending_by_qubit[q] = _SZZ_FLOW_IDENTITY

    def discharge_for_two_qubit(q: int, host_index: int, op: SurfaceCircuitStep) -> None:
        nonlocal physical_prefix_pulses, two_qubit_prefix_pulses, virtual_z_two_qubit_carries
        current = pending_for(q)
        if current == _SZZ_FLOW_IDENTITY:
            return
        if _szz_flow_is_virtual_z(current):
            virtual_z_two_qubit_carries += 1
            pulses.append(
                pulse_event(
                    host_index=host_index,
                    op=op,
                    qubit=q,
                    kind="virtual_z_two_qubit_carry",
                    pending=current,
                ),
            )
            return
        physical_prefix_pulses += 1
        two_qubit_prefix_pulses += 1
        pulses.append(
            pulse_event(
                host_index=host_index,
                op=op,
                qubit=q,
                kind="physical_two_qubit_prefix",
                pending=current,
            ),
        )
        pending_by_qubit[q] = _SZZ_FLOW_IDENTITY

    def discharge_for_measurement(q: int, host_index: int, op: SurfaceCircuitStep) -> None:
        nonlocal physical_prefix_pulses, measurement_prefix_pulses, virtual_z_measure_discards
        current = pending_for(q)
        if current == _SZZ_FLOW_IDENTITY:
            return
        if _szz_flow_is_virtual_z(current):
            virtual_z_measure_discards += 1
            pulses.append(
                pulse_event(
                    host_index=host_index,
                    op=op,
                    qubit=q,
                    kind="virtual_z_measure_discard",
                    pending=current,
                ),
            )
            pending_by_qubit[q] = _SZZ_FLOW_IDENTITY
            return
        physical_prefix_pulses += 1
        measurement_prefix_pulses += 1
        pulses.append(
            pulse_event(
                host_index=host_index,
                op=op,
                qubit=q,
                kind="physical_measurement_prefix",
                pending=current,
            ),
        )
        pending_by_qubit[q] = _SZZ_FLOW_IDENTITY

    for host_index, op in enumerate(ops):
        if op.op_type in {OpType.COMMENT, OpType.TICK, OpType.TRACKED_PAULI}:
            continue
        if op.op_type in {OpType.ALLOC, OpType.PREP}:
            prep_events += 1
            reset_for_prep(op.qubits[0], op)
            continue
        if op.op_type in _SZZ_FLOW_SINGLE_QUBIT_GATES:
            q = op.qubits[0]
            abstract_single_qubit_ops += 1
            pending_by_qubit[q] = _szz_flow_compose_pending_gate(pending_for(q), op.op_type)
            continue
        if op.op_type in {OpType.SZZ, OpType.SZZDG}:
            two_qubit_gates += 1
            for q in op.qubits:
                discharge_for_two_qubit(q, host_index, op)
            continue
        if op.op_type == OpType.CX:
            msg = "SZZ forward-flow analysis only supports SZZ/SZZdg two-qubit gates"
            raise ValueError(msg)
        if op.op_type == OpType.MEASURE:
            measurements += 1
            discharge_for_measurement(op.qubits[0], host_index, op)
            continue

    remaining = {q: pending for q, pending in pending_by_qubit.items() if pending != _SZZ_FLOW_IDENTITY}
    if remaining:
        formatted = {q: _szz_flow_clifford_name(pending) for q, pending in sorted(remaining.items())}
        msg = f"SZZ forward-flow ended with pending Cliffords: {formatted}"
        raise ValueError(msg)

    return SzzForwardFlowSummary(
        abstract_single_qubit_ops=abstract_single_qubit_ops,
        physical_prefix_pulses=physical_prefix_pulses,
        two_qubit_prefix_pulses=two_qubit_prefix_pulses,
        measurement_prefix_pulses=measurement_prefix_pulses,
        virtual_z_two_qubit_carries=virtual_z_two_qubit_carries,
        virtual_z_measure_discards=virtual_z_measure_discards,
        two_qubit_gates=two_qubit_gates,
        measurements=measurements,
        prep_events=prep_events,
        free_standing_single_qubit_ops=0,
        pulses=tuple(pulses),
    )


_SZZ_FLOW_PHYSICAL_PREFIX_BY_PENDING: dict[tuple[int, int], tuple[OpType | None, OpType]] = {
    (3, 1): (None, OpType.H),
    (-3, 1): (None, OpType.SY),
    (2, 1): (None, OpType.F),
    (-2, 1): (OpType.Z, OpType.F),
    (1, 2): (None, OpType.SXDG),
    (-1, 2): (OpType.Z, OpType.SXDG),
    (3, 2): (None, OpType.FDG),
    (-3, 2): (OpType.Z, OpType.FDG),
}


def _lower_szz_forward_flow_ops(ops: list[SurfaceCircuitStep]) -> list[SurfaceCircuitStep]:
    """Return an SZZ physical-prefix TickCircuit op stream.

    Free-standing single-qubit Cliffords in the abstract SZZ template are
    accumulated into a pending local Clifford and discharged as a physical
    prefix pulse on the next SZZ/SZZdg or MZ host. Zero-noise Z-frame gates are
    emitted only when needed to preserve the exact signed Clifford before a
    physical prefix pulse.
    """
    pending_by_qubit: dict[int, tuple[int, int]] = {}
    lowered: list[SurfaceCircuitStep] = []

    def pending_for(q: int) -> tuple[int, int]:
        return pending_by_qubit.setdefault(q, _SZZ_FLOW_IDENTITY)

    def reset_for_prep(q: int, op: SurfaceCircuitStep) -> None:
        current = pending_for(q)
        if current != _SZZ_FLOW_IDENTITY:
            msg = (
                "SZZ forward-flow cannot reset a qubit with a pending "
                f"Clifford: q={q}, pending={_szz_flow_clifford_name(current)}, "
                f"op={op.op_type.name} {op.label!r}"
            )
            raise ValueError(msg)
        pending_by_qubit[q] = _SZZ_FLOW_IDENTITY

    def discharge(q: int, host: SurfaceCircuitStep) -> tuple[SurfaceCircuitStep | None, SurfaceCircuitStep | None]:
        current = pending_for(q)
        if current == _SZZ_FLOW_IDENTITY:
            return None, None
        if _szz_flow_is_virtual_z(current):
            pending_by_qubit[q] = _SZZ_FLOW_IDENTITY if host.op_type == OpType.MEASURE else current
            return None, None
        try:
            virtual_gate, physical_gate = _SZZ_FLOW_PHYSICAL_PREFIX_BY_PENDING[current]
        except KeyError as exc:
            msg = (
                "SZZ forward-flow cannot lower pending Clifford "
                f"{_szz_flow_clifford_name(current)} on q={q} before "
                f"{host.op_type.name} {host.label!r}"
            )
            raise ValueError(msg) from exc
        virtual_step = None
        if virtual_gate is not None:
            virtual_step = SurfaceCircuitStep(
                virtual_gate,
                [q],
                f"szz_virtual_prefix:{virtual_gate.name}:{host.label}:q{q}",
            )
        physical_step = SurfaceCircuitStep(
            physical_gate,
            [q],
            f"szz_physical_prefix:{physical_gate.name}:{host.label}:q{q}",
        )
        pending_by_qubit[q] = _SZZ_FLOW_IDENTITY
        return virtual_step, physical_step

    def append_prefix_ticks(virtual_steps: list[SurfaceCircuitStep], physical_steps: list[SurfaceCircuitStep]) -> None:
        if virtual_steps:
            lowered.append(SurfaceCircuitStep(OpType.TICK))
            lowered.extend(virtual_steps)
            lowered.append(SurfaceCircuitStep(OpType.TICK))
        if physical_steps:
            lowered.append(SurfaceCircuitStep(OpType.TICK))
            lowered.extend(physical_steps)
            lowered.append(SurfaceCircuitStep(OpType.TICK))

    for op in ops:
        if op.op_type in {OpType.COMMENT, OpType.TICK, OpType.TRACKED_PAULI}:
            lowered.append(op)
            continue
        if op.op_type in {OpType.ALLOC, OpType.PREP}:
            reset_for_prep(op.qubits[0], op)
            lowered.append(op)
            continue
        if op.op_type in _SZZ_FLOW_SINGLE_QUBIT_GATES:
            q = op.qubits[0]
            pending_by_qubit[q] = _szz_flow_compose_pending_gate(pending_for(q), op.op_type)
            continue
        if op.op_type in {OpType.SZZ, OpType.SZZDG}:
            virtual_steps: list[SurfaceCircuitStep] = []
            physical_steps: list[SurfaceCircuitStep] = []
            for q in op.qubits:
                virtual_step, physical_step = discharge(q, op)
                if virtual_step is not None:
                    virtual_steps.append(virtual_step)
                if physical_step is not None:
                    physical_steps.append(physical_step)
            append_prefix_ticks(virtual_steps, physical_steps)
            lowered.append(op)
            continue
        if op.op_type == OpType.CX:
            msg = "SZZ forward-flow lowering only supports SZZ/SZZdg two-qubit gates"
            raise ValueError(msg)
        if op.op_type == OpType.MEASURE:
            virtual_step, physical_step = discharge(op.qubits[0], op)
            append_prefix_ticks(
                [] if virtual_step is None else [virtual_step],
                [] if physical_step is None else [physical_step],
            )
            lowered.append(op)
            continue
        lowered.append(op)

    remaining = {q: pending for q, pending in pending_by_qubit.items() if pending != _SZZ_FLOW_IDENTITY}
    if remaining:
        formatted = {q: _szz_flow_clifford_name(pending) for q, pending in sorted(remaining.items())}
        msg = f"SZZ forward-flow lowering ended with pending Cliffords: {formatted}"
        raise ValueError(msg)
    return lowered


def build_surface_code_circuit(
    patch: SurfacePatch,
    num_rounds: int,
    basis: str = "Z",
    ancilla_budget: int | None = None,
    *,
    twirl: TwirlConfig | None = None,
    interaction_basis: str | None = None,
    check_plan: str | None = None,
    clifford_frame_policy: str | None = None,
) -> tuple[list[SurfaceCircuitStep], QubitAllocation]:
    """Build abstract circuit operations for a surface code memory experiment.

    This generates the circuit structure matching the Guppy implementation:
    1. prep_{basis}_basis: Allocate and prepare data qubits
    2. init syndrome establishment for the random-sign stabilizer family
    3. syndrome_extraction x num_rounds: Syndrome extraction with fresh ancillas
    4. measure_{basis}_basis: Final data qubit measurement

    Args:
        patch: Surface code patch with geometry
        num_rounds: Number of syndrome extraction rounds
        basis: 'Z' for |0_L> state or 'X' for |+_L> state
        ancilla_budget: Optional cap on simultaneously live ancillas. When
            provided below the total stabilizer count, ancillas are reused
            across stabilizer batches following the public Guppy order.
        twirl: When provided, emit three ``OpType.TRACKED_PAULI`` annotations
            (``X``, ``Y``, ``Z``) per Pauli-mask column. The
            ``"between_rounds"`` schedule emits one column per data qubit at
            each site between counted syndrome rounds. The
            ``"before_two_qubit_gate"`` schedule emits one column per operand
            immediately before each surface-memory two-qubit gate.
        interaction_basis: Surface-memory two-qubit interaction basis.
            ``"cx"`` preserves the existing CNOT extraction circuit. ``"szz"``
            emits the direct-renderer SZZ/SZZdg abstract template with local
            data-qubit compensation.
        check_plan: Named surface check-plan preset. This is the source of
            truth when supplied; ``interaction_basis`` must agree if also
            supplied.
        clifford_frame_policy: Optional source-level Clifford-deformation
            policy. Currently supported only by the SZZ renderer. Global
            axis-cycle frames and checkerboard XZZX/ZXXZ frames are rendered
            as concrete deformed checks.

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
    resolved_plan = resolve_surface_check_plan(
        interaction_basis=interaction_basis,
        check_plan=check_plan,
    )
    require_current_surface_check_plan_renderer(
        resolved_plan,
        context="abstract surface-code circuit generation",
    )
    ancilla_schedule = ancilla_schedule_for_check_plan(resolved_plan)
    cnot_round_order = cnot_round_order_for_check_plan(resolved_plan)
    interaction_basis = _normalize_interaction_basis(resolved_plan.interaction_basis)
    resolved_clifford_frame = _resolve_szz_clifford_frame_for_builder(
        patch,
        interaction_basis=interaction_basis,
        clifford_frame_policy=clifford_frame_policy,
    )
    if twirl is not None:
        twirl.validate_runtime_supported()
    twirl_site_schedule = None if twirl is None else twirl.site_schedule

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
        for batch in _batched_stabilizers(
            patch,
            effective_ancilla_budget,
            ancilla_schedule=ancilla_schedule,
        ):
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

    def emit_between_round_twirl_site(target_ops: list[SurfaceCircuitStep], site_idx: int) -> None:
        """Append 3 * num_data candidate tracked-Pauli annotations."""
        target_ops.extend(
            SurfaceCircuitStep(OpType.TRACKED_PAULI, [data_q(i)], f"{kind}@s{site_idx}")
            for i in range(num_data)
            for kind in ("X", "Y", "Z")
        )

    gate_twirl_site_idx = 0

    def emit_gate_local_twirl_site(
        target_ops: list[SurfaceCircuitStep],
        site_idx: int,
        control_q: int,
        target_q: int,
    ) -> None:
        """Append tracked-Pauli annotations for one two-qubit-gate site."""
        for operand_idx, q in enumerate((control_q, target_q)):
            target_ops.extend(
                SurfaceCircuitStep(
                    OpType.TRACKED_PAULI,
                    [q],
                    f"{kind}@g{site_idx}o{operand_idx}",
                )
                for kind in ("X", "Y", "Z")
            )

    def emit_gate_local_twirl_layer(
        target_ops: list[SurfaceCircuitStep],
        cx_ops: list[tuple[int, int, str]],
    ) -> None:
        """Append all gate-local twirl annotations before a parallel CX layer."""
        nonlocal gate_twirl_site_idx
        if twirl_site_schedule != "before_two_qubit_gate":
            return
        for control_q, target_q, _label in cx_ops:
            emit_gate_local_twirl_site(
                target_ops,
                gate_twirl_site_idx,
                control_q,
                target_q,
            )
            gate_twirl_site_idx += 1

    # Get CNOT schedule
    cnot_rounds = compute_cnot_schedule(patch, round_order=cnot_round_order)

    if interaction_basis == "szz":
        if twirl is not None:
            msg = "interaction_basis='szz' twirl integration is staged later; omit twirl for Stage 1"
            raise ValueError(msg)
        return (
            _build_surface_code_circuit_szz(
                patch,
                num_rounds,
                basis,
                allocation,
                cnot_rounds,
                _szz_residual_plan_for_check_plan(patch, resolved_plan),
                ancilla_schedule,
                resolved_clifford_frame,
            ),
            allocation,
        )

    ops: list[SurfaceCircuitStep] = []

    # =========================================================================
    # prep_z_basis / prep_x_basis
    # =========================================================================
    ops.append(SurfaceCircuitStep(OpType.COMMENT, label=f"prep_{basis.lower()}_basis"))

    # Allocate and reset data qubits
    ops.extend(SurfaceCircuitStep(OpType.ALLOC, [data_q(i)], f"data[{i}]") for i in range(num_data))

    # For X-basis: H on each data qubit
    if basis.upper() == "X":
        ops.extend(SurfaceCircuitStep(OpType.H, [data_q(i)]) for i in range(num_data))

    ops.append(SurfaceCircuitStep(OpType.TICK))

    # =========================================================================
    # init_{basis}_basis syndrome establishment
    # =========================================================================
    # Data prep fixes only the stabilizers matching the memory basis. Measure
    # the complementary stabilizer family once to establish its random signs;
    # this is logical state prep and is intentionally not counted in
    # `num_rounds`.
    init_stabilizer_type = "X" if basis.upper() == "Z" else "Z"
    ops.append(
        SurfaceCircuitStep(
            OpType.COMMENT,
            label=f"init_{init_stabilizer_type.lower()}_syndrome",
        ),
    )
    if effective_ancilla_budget == total_ancilla:
        init_stabilizers = geom.x_stabilizers if init_stabilizer_type == "X" else geom.z_stabilizers
        init_anc_q = x_anc_q if init_stabilizer_type == "X" else z_anc_q

        ops.extend(
            SurfaceCircuitStep(
                OpType.ALLOC,
                [init_anc_q(s.index)],
                f"a{init_stabilizer_type.lower()}{s.index}",
            )
            for s in init_stabilizers
        )

        if init_stabilizer_type == "X":
            ops.append(SurfaceCircuitStep(OpType.COMMENT, label="Hadamard on X ancillas"))
            ops.extend(SurfaceCircuitStep(OpType.H, [x_anc_q(s.index)], f"ax{s.index}") for s in init_stabilizers)

        ops.append(SurfaceCircuitStep(OpType.TICK))

        for rnd_idx, cx_round in enumerate(cnot_rounds):
            ops.append(SurfaceCircuitStep(OpType.COMMENT, label=f"CX round {rnd_idx + 1}"))
            cx_ops: list[tuple[int, int, str]] = []
            for stab_type, stab_idx, data_idx in cx_round:
                if stab_type != init_stabilizer_type:
                    continue
                if stab_type == "X":
                    cx_ops.append((x_anc_q(stab_idx), data_q(data_idx), f"X{stab_idx}"))
                else:
                    cx_ops.append((data_q(data_idx), z_anc_q(stab_idx), f"Z{stab_idx}"))
            emit_gate_local_twirl_layer(ops, cx_ops)
            ops.extend(SurfaceCircuitStep(OpType.CX, [control, target], label) for control, target, label in cx_ops)
            ops.append(SurfaceCircuitStep(OpType.TICK))

        if init_stabilizer_type == "X":
            ops.append(SurfaceCircuitStep(OpType.COMMENT, label="Hadamard on X ancillas"))
            ops.extend(SurfaceCircuitStep(OpType.H, [x_anc_q(s.index)], f"ax{s.index}") for s in init_stabilizers)

        ops.append(SurfaceCircuitStep(OpType.COMMENT, label="Measure ancillas"))
        init_label_prefix = "sx" if init_stabilizer_type == "X" else "sz"
        ops.extend(
            SurfaceCircuitStep(
                OpType.MEASURE,
                [init_anc_q(s.index)],
                f"{init_label_prefix}{s.index}",
            )
            for s in init_stabilizers
        )

        ops.append(SurfaceCircuitStep(OpType.TICK))
    else:
        stabilizer_batches = _batched_stabilizers(
            patch,
            effective_ancilla_budget,
            ancilla_schedule=ancilla_schedule,
        )
        for batch in stabilizer_batches:
            init_batch = [(stab_type, stab_idx) for stab_type, stab_idx in batch if stab_type == init_stabilizer_type]
            if not init_batch:
                continue
            ops.append(SurfaceCircuitStep(OpType.COMMENT, label="Prepare ancillas"))
            batch_ancillas = {
                (stab_type, stab_idx): x_anc_q(stab_idx) if stab_type == "X" else z_anc_q(stab_idx)
                for stab_type, stab_idx in init_batch
            }

            for stab_type, stab_idx in init_batch:
                ops.append(
                    SurfaceCircuitStep(
                        OpType.ALLOC,
                        [batch_ancillas[(stab_type, stab_idx)]],
                        f"a{stab_type.lower()}{stab_idx}",
                    ),
                )

            if init_stabilizer_type == "X":
                ops.append(SurfaceCircuitStep(OpType.COMMENT, label="Hadamard on X ancillas"))
                ops.extend(
                    SurfaceCircuitStep(OpType.H, [batch_ancillas[("X", stab_idx)]], f"ax{stab_idx}")
                    for _stab_type, stab_idx in init_batch
                )

            ops.append(SurfaceCircuitStep(OpType.TICK))

            for rnd_idx, cx_round in enumerate(cnot_rounds):
                ops.append(SurfaceCircuitStep(OpType.COMMENT, label=f"CX round {rnd_idx + 1}"))
                cx_ops: list[tuple[int, int, str]] = []
                for stab_type, stab_idx, data_idx in cx_round:
                    ancilla_q = batch_ancillas.get((stab_type, stab_idx))
                    if ancilla_q is None:
                        continue
                    if stab_type == "X":
                        cx_ops.append((ancilla_q, data_q(data_idx), f"X{stab_idx}"))
                    else:
                        cx_ops.append((data_q(data_idx), ancilla_q, f"Z{stab_idx}"))
                emit_gate_local_twirl_layer(ops, cx_ops)
                ops.extend(SurfaceCircuitStep(OpType.CX, [control, target], label) for control, target, label in cx_ops)
                ops.append(SurfaceCircuitStep(OpType.TICK))

            if init_stabilizer_type == "X":
                ops.append(SurfaceCircuitStep(OpType.COMMENT, label="Hadamard on X ancillas"))
                ops.extend(
                    SurfaceCircuitStep(OpType.H, [batch_ancillas[("X", stab_idx)]], f"ax{stab_idx}")
                    for _stab_type, stab_idx in init_batch
                )

            ops.append(SurfaceCircuitStep(OpType.COMMENT, label="Measure ancillas"))
            for stab_type, stab_idx in init_batch:
                measure_label = f"sx{stab_idx}" if stab_type == "X" else f"sz{stab_idx}"
                ops.append(
                    SurfaceCircuitStep(
                        OpType.MEASURE,
                        [batch_ancillas[(stab_type, stab_idx)]],
                        measure_label,
                    ),
                )

            ops.append(SurfaceCircuitStep(OpType.TICK))

    # =========================================================================
    # syndrome_extraction (called num_rounds times)
    # =========================================================================
    for rnd in range(num_rounds):
        ops.append(
            SurfaceCircuitStep(OpType.COMMENT, label=f"syndrome_extraction round {rnd + 1}"),
        )
        if effective_ancilla_budget == total_ancilla:
            ops.extend(SurfaceCircuitStep(OpType.ALLOC, [x_anc_q(s.index)], f"ax{s.index}") for s in geom.x_stabilizers)
            ops.extend(SurfaceCircuitStep(OpType.ALLOC, [z_anc_q(s.index)], f"az{s.index}") for s in geom.z_stabilizers)

            ops.append(SurfaceCircuitStep(OpType.COMMENT, label="Hadamard on X ancillas"))
            ops.extend(SurfaceCircuitStep(OpType.H, [x_anc_q(s.index)], f"ax{s.index}") for s in geom.x_stabilizers)

            ops.append(SurfaceCircuitStep(OpType.TICK))

            for rnd_idx, cx_round in enumerate(cnot_rounds):
                ops.append(SurfaceCircuitStep(OpType.COMMENT, label=f"CX round {rnd_idx + 1}"))
                cx_ops: list[tuple[int, int, str]] = []
                for stab_type, stab_idx, data_idx in cx_round:
                    if stab_type == "X":
                        cx_ops.append((x_anc_q(stab_idx), data_q(data_idx), f"X{stab_idx}"))
                    else:
                        cx_ops.append((data_q(data_idx), z_anc_q(stab_idx), f"Z{stab_idx}"))
                emit_gate_local_twirl_layer(ops, cx_ops)
                ops.extend(SurfaceCircuitStep(OpType.CX, [control, target], label) for control, target, label in cx_ops)
                ops.append(SurfaceCircuitStep(OpType.TICK))

            ops.append(SurfaceCircuitStep(OpType.COMMENT, label="Hadamard on X ancillas"))
            ops.extend(SurfaceCircuitStep(OpType.H, [x_anc_q(s.index)], f"ax{s.index}") for s in geom.x_stabilizers)

            ops.append(SurfaceCircuitStep(OpType.COMMENT, label="Measure ancillas"))
            ops.extend(
                SurfaceCircuitStep(OpType.MEASURE, [x_anc_q(s.index)], f"sx{s.index}") for s in geom.x_stabilizers
            )
            ops.extend(
                SurfaceCircuitStep(OpType.MEASURE, [z_anc_q(s.index)], f"sz{s.index}") for s in geom.z_stabilizers
            )

            ops.append(SurfaceCircuitStep(OpType.TICK))
        else:
            stabilizer_batches = _batched_stabilizers(
                patch,
                effective_ancilla_budget,
                ancilla_schedule=ancilla_schedule,
            )
            for batch in stabilizer_batches:
                ops.append(SurfaceCircuitStep(OpType.COMMENT, label="Prepare ancillas"))
                batch_ancillas = {
                    (stab_type, stab_idx): x_anc_q(stab_idx) if stab_type == "X" else z_anc_q(stab_idx)
                    for stab_type, stab_idx in batch
                }

                for stab_type, stab_idx in batch:
                    ops.append(
                        SurfaceCircuitStep(
                            OpType.ALLOC,
                            [batch_ancillas[(stab_type, stab_idx)]],
                            f"a{stab_type.lower()}{stab_idx}",
                        ),
                    )

                x_stabilizers_in_batch = [stab_idx for stab_type, stab_idx in batch if stab_type == "X"]
                if x_stabilizers_in_batch:
                    ops.append(SurfaceCircuitStep(OpType.COMMENT, label="Hadamard on X ancillas"))
                    ops.extend(
                        SurfaceCircuitStep(OpType.H, [batch_ancillas[("X", stab_idx)]], f"ax{stab_idx}")
                        for stab_idx in x_stabilizers_in_batch
                    )

                ops.append(SurfaceCircuitStep(OpType.TICK))

                for rnd_idx, cx_round in enumerate(cnot_rounds):
                    ops.append(SurfaceCircuitStep(OpType.COMMENT, label=f"CX round {rnd_idx + 1}"))
                    cx_ops: list[tuple[int, int, str]] = []
                    for stab_type, stab_idx, data_idx in cx_round:
                        ancilla_q = batch_ancillas.get((stab_type, stab_idx))
                        if ancilla_q is None:
                            continue
                        if stab_type == "X":
                            cx_ops.append((ancilla_q, data_q(data_idx), f"X{stab_idx}"))
                        else:
                            cx_ops.append((data_q(data_idx), ancilla_q, f"Z{stab_idx}"))
                    emit_gate_local_twirl_layer(ops, cx_ops)
                    ops.extend(
                        SurfaceCircuitStep(OpType.CX, [control, target], label) for control, target, label in cx_ops
                    )
                    ops.append(SurfaceCircuitStep(OpType.TICK))

                if x_stabilizers_in_batch:
                    ops.append(SurfaceCircuitStep(OpType.COMMENT, label="Hadamard on X ancillas"))
                    ops.extend(
                        SurfaceCircuitStep(OpType.H, [batch_ancillas[("X", stab_idx)]], f"ax{stab_idx}")
                        for stab_idx in x_stabilizers_in_batch
                    )

                ops.append(SurfaceCircuitStep(OpType.COMMENT, label="Measure ancillas"))
                for stab_type, stab_idx in batch:
                    measure_label = f"sx{stab_idx}" if stab_type == "X" else f"sz{stab_idx}"
                    ops.append(
                        SurfaceCircuitStep(
                            OpType.MEASURE,
                            [batch_ancillas[(stab_type, stab_idx)]],
                            measure_label,
                        ),
                    )

                ops.append(SurfaceCircuitStep(OpType.TICK))

        if twirl_site_schedule == "between_rounds" and rnd < num_rounds - 1:
            emit_between_round_twirl_site(ops, rnd)

    # =========================================================================
    # measure_z_basis / measure_x_basis
    # =========================================================================
    ops.append(SurfaceCircuitStep(OpType.COMMENT, label=f"measure_{basis.lower()}_basis"))

    # For X-basis: H on each data qubit first
    if basis.upper() == "X":
        ops.extend(SurfaceCircuitStep(OpType.H, [data_q(i)]) for i in range(num_data))

    # Measure all data qubits
    ops.extend(SurfaceCircuitStep(OpType.MEASURE, [data_q(i)], f"final[{i}]") for i in range(num_data))

    return ops, allocation


def _build_surface_code_circuit_szz(
    patch: SurfacePatch,
    num_rounds: int,
    basis: str,
    allocation: QubitAllocation,
    cnot_rounds: list[list[tuple[str, int, int]]],
    residual_plan: SzzResidualPlan,
    ancilla_schedule: str,
    resolved_clifford_frame: ResolvedSurfaceCliffordFrame | None = None,
) -> list[SurfaceCircuitStep]:
    """Build the abstract SZZ/SZZdg surface-memory template."""
    geom = patch.geometry
    num_data = geom.num_data
    check_by_key: dict[tuple[str, int], ResolvedPauliCheck] = {}
    if resolved_clifford_frame is not None:
        check_by_key.update({("X", check.stabilizer_index): check for check in resolved_clifford_frame.x_checks})
        check_by_key.update({("Z", check.stabilizer_index): check for check in resolved_clifford_frame.z_checks})
    sign_by_touch = {
        (entry.stabilizer_type, entry.stabilizer_index, entry.data_qubit): entry.sign for entry in residual_plan.signs
    }
    gate_name_by_type = {
        OpType.SX: "SX",
        OpType.SXDG: "SXDG",
        OpType.SY: "SY",
        OpType.SYDG: "SYDG",
        OpType.SZ: "SZ",
        OpType.SZDG: "SZDG",
    }

    def data_q(i: int) -> int:
        return allocation.data_qubits[i]

    def x_anc_q(stab_idx: int) -> int:
        return allocation.x_ancilla_qubits[stab_idx]

    def z_anc_q(stab_idx: int) -> int:
        return allocation.z_ancilla_qubits[stab_idx]

    def anc_q(stabilizer_type: str, stab_idx: int) -> int:
        return x_anc_q(stab_idx) if stabilizer_type == "X" else z_anc_q(stab_idx)

    def physical_axis_for_touch(stabilizer_type: str, stab_idx: int, data_idx: int) -> PauliAxis:
        if resolved_clifford_frame is None:
            return "X" if stabilizer_type == "X" else "Z"
        check = check_by_key[(stabilizer_type, stab_idx)]
        try:
            offset = check.data_qubits.index(data_idx)
        except ValueError as exc:
            msg = f"data qubit {data_idx} is not in resolved check {stabilizer_type}{stab_idx}"
            raise ValueError(msg) from exc
        return check.paulis[offset].axis

    def physical_axis_for_memory_data(data_idx: int) -> PauliAxis:
        return _szz_memory_physical_axis_for_data(
            basis,
            resolved_clifford_frame,
            data_idx,
        )

    def append_axis_rotation_to_z(
        target_ops: list[SurfaceCircuitStep],
        axis: PauliAxis,
        qubit: int,
        label_prefix: str,
    ) -> None:
        gate = {
            "X": OpType.H,
            "Y": OpType.SXDG,
            "Z": None,
        }[axis]
        if gate is not None:
            target_ops.append(SurfaceCircuitStep(gate, [qubit], f"{label_prefix}:to_z"))

    def append_axis_rotation_from_z(
        target_ops: list[SurfaceCircuitStep],
        axis: PauliAxis,
        qubit: int,
        label_prefix: str,
    ) -> None:
        gate = {
            "X": OpType.H,
            "Y": OpType.SX,
            "Z": None,
        }[axis]
        if gate is not None:
            target_ops.append(SurfaceCircuitStep(gate, [qubit], f"{label_prefix}:from_z"))

    def szz_touch_compensation(axis: PauliAxis, sign: int) -> OpType:
        return {
            ("X", 1): OpType.SXDG,
            ("X", -1): OpType.SX,
            ("Y", 1): OpType.SYDG,
            ("Y", -1): OpType.SY,
            ("Z", 1): OpType.SZDG,
            ("Z", -1): OpType.SZ,
        }[(axis, sign)]

    def stabilizer_batches_for(selected_type: str | None = None) -> list[list[tuple[str, int]]]:
        """Return the same ancilla-reuse batches used by the CX template."""
        total_ancilla = len(geom.x_stabilizers) + len(geom.z_stabilizers)
        allocation_ancilla_count = len(set(allocation.x_ancilla_qubits + allocation.z_ancilla_qubits))
        if allocation_ancilla_count >= total_ancilla:
            batch = [("X", s.index) for s in geom.x_stabilizers]
            batch.extend(("Z", s.index) for s in geom.z_stabilizers)
            batches = [batch]
        else:
            batches = _batched_stabilizers(
                patch,
                allocation_ancilla_count,
                ancilla_schedule=ancilla_schedule,
            )

        if selected_type is None:
            return batches
        return [
            [(stab_type, stab_idx) for stab_type, stab_idx in batch if stab_type == selected_type]
            for batch in batches
            if any(stab_type == selected_type for stab_type, _stab_idx in batch)
        ]

    def append_prepare_szz_ancillas(target_ops: list[SurfaceCircuitStep], batch: list[tuple[str, int]]) -> None:
        target_ops.extend(
            SurfaceCircuitStep(
                OpType.ALLOC,
                [anc_q(stab_type, stab_idx)],
                f"a{stab_type.lower()}{stab_idx}",
            )
            for stab_type, stab_idx in batch
        )
        target_ops.append(SurfaceCircuitStep(OpType.COMMENT, label="Hadamard on SZZ ancillas"))
        target_ops.extend(
            SurfaceCircuitStep(
                OpType.H,
                [anc_q(stab_type, stab_idx)],
                f"a{stab_type.lower()}{stab_idx}",
            )
            for stab_type, stab_idx in batch
        )

    def append_measure_szz_ancillas(target_ops: list[SurfaceCircuitStep], batch: list[tuple[str, int]]) -> None:
        target_ops.append(SurfaceCircuitStep(OpType.COMMENT, label="Hadamard on SZZ ancillas"))
        target_ops.extend(
            SurfaceCircuitStep(
                OpType.H,
                [anc_q(stab_type, stab_idx)],
                f"a{stab_type.lower()}{stab_idx}",
            )
            for stab_type, stab_idx in batch
        )

        target_ops.append(SurfaceCircuitStep(OpType.COMMENT, label="Measure ancillas"))
        target_ops.extend(
            SurfaceCircuitStep(
                OpType.MEASURE,
                [anc_q(stab_type, stab_idx)],
                f"{'sx' if stab_type == 'X' else 'sz'}{stab_idx}",
            )
            for stab_type, stab_idx in batch
        )

    def append_szz_layer(
        target_ops: list[SurfaceCircuitStep],
        rnd_idx: int,
        layer_gates: list[tuple[str, int, int]],
    ) -> None:
        target_ops.append(SurfaceCircuitStep(OpType.COMMENT, label=f"SZZ round {rnd_idx + 1}"))
        for stab_type, stab_idx, data_idx in layer_gates:
            axis = physical_axis_for_touch(stab_type, stab_idx, data_idx)
            append_axis_rotation_to_z(
                target_ops,
                axis,
                data_q(data_idx),
                f"szz_{axis.lower()}_touch_pre:{stab_type}{stab_idx}:d{data_idx}",
            )

        for stab_type, stab_idx, data_idx in layer_gates:
            sign = sign_by_touch[(stab_type, stab_idx, data_idx)]
            op_type = OpType.SZZ if sign > 0 else OpType.SZZDG
            target_ops.append(
                SurfaceCircuitStep(
                    op_type,
                    [anc_q(stab_type, stab_idx), data_q(data_idx)],
                    f"{stab_type}{stab_idx}",
                ),
            )

        for stab_type, stab_idx, data_idx in layer_gates:
            axis = physical_axis_for_touch(stab_type, stab_idx, data_idx)
            append_axis_rotation_from_z(
                target_ops,
                axis,
                data_q(data_idx),
                f"szz_{axis.lower()}_touch_post:{stab_type}{stab_idx}:d{data_idx}",
            )

        for stab_type, stab_idx, data_idx in layer_gates:
            axis = physical_axis_for_touch(stab_type, stab_idx, data_idx)
            sign = sign_by_touch[(stab_type, stab_idx, data_idx)]
            compensation = szz_touch_compensation(axis, sign)
            target_ops.append(
                SurfaceCircuitStep(
                    compensation,
                    [data_q(data_idx)],
                    f"szz_touch_comp:{gate_name_by_type[compensation]}:{axis}:{stab_type}{stab_idx}:d{data_idx}",
                ),
            )

    ops: list[SurfaceCircuitStep] = []

    # =========================================================================
    # prep_z_basis / prep_x_basis
    # =========================================================================
    ops.append(SurfaceCircuitStep(OpType.COMMENT, label=f"prep_{basis.lower()}_basis"))
    ops.extend(SurfaceCircuitStep(OpType.ALLOC, [data_q(i)], f"data[{i}]") for i in range(num_data))
    for i in range(num_data):
        basis_axis = physical_axis_for_memory_data(i)
        append_axis_rotation_to_z(
            ops,
            basis_axis,
            data_q(i),
            f"prep_{basis_axis.lower()}_basis_d{i}",
        )
    ops.append(SurfaceCircuitStep(OpType.TICK))

    # =========================================================================
    # init_{basis}_basis syndrome establishment
    # =========================================================================
    init_stabilizer_type = "X" if basis.upper() == "Z" else "Z"
    ops.append(
        SurfaceCircuitStep(
            OpType.COMMENT,
            label=f"init_{init_stabilizer_type.lower()}_syndrome",
        ),
    )
    for init_batch in stabilizer_batches_for(init_stabilizer_type):
        append_prepare_szz_ancillas(ops, init_batch)
        ops.append(SurfaceCircuitStep(OpType.TICK))

        init_keys = set(init_batch)
        for rnd_idx, cnot_round in enumerate(cnot_rounds):
            layer_gates = [
                (stab_type, stab_idx, data_idx)
                for stab_type, stab_idx, data_idx in cnot_round
                if (stab_type, stab_idx) in init_keys
            ]
            append_szz_layer(ops, rnd_idx, layer_gates)
            ops.append(SurfaceCircuitStep(OpType.TICK))

        append_measure_szz_ancillas(ops, init_batch)
        ops.append(SurfaceCircuitStep(OpType.TICK))

    # =========================================================================
    # syndrome_extraction
    # =========================================================================
    stabilizer_batches = stabilizer_batches_for()
    for rnd in range(num_rounds):
        ops.append(
            SurfaceCircuitStep(OpType.COMMENT, label=f"syndrome_extraction round {rnd + 1}"),
        )

        for batch in stabilizer_batches:
            append_prepare_szz_ancillas(ops, batch)
            ops.append(SurfaceCircuitStep(OpType.TICK))

            batch_keys = set(batch)
            for rnd_idx, cnot_round in enumerate(cnot_rounds):
                layer_gates = [
                    (stab_type, stab_idx, data_idx)
                    for stab_type, stab_idx, data_idx in cnot_round
                    if (stab_type, stab_idx) in batch_keys
                ]
                append_szz_layer(ops, rnd_idx, layer_gates)
                ops.append(SurfaceCircuitStep(OpType.TICK))

            append_measure_szz_ancillas(ops, batch)
            ops.append(SurfaceCircuitStep(OpType.TICK))

    # =========================================================================
    # measure_z_basis / measure_x_basis
    # =========================================================================
    ops.append(SurfaceCircuitStep(OpType.COMMENT, label=f"measure_{basis.lower()}_basis"))
    for i in range(num_data):
        basis_axis = physical_axis_for_memory_data(i)
        append_axis_rotation_from_z(
            ops,
            basis_axis,
            data_q(i),
            f"measure_{basis_axis.lower()}_basis_d{i}",
        )
    ops.extend(SurfaceCircuitStep(OpType.MEASURE, [data_q(i)], f"final[{i}]") for i in range(num_data))

    _analyze_szz_forward_flow(ops)
    return ops


def classify_stabilizer_boundary(stab_type: str, data_qubits: tuple[int, ...], d: int, dz: int | None = None) -> str:
    """Public wrapper for classifying a boundary stabilizer."""
    from pecos.qec.surface.schedule import _classify_boundary

    if dz is None:
        dz = d
    return _classify_boundary(stab_type, data_qubits, d, dz)


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
        ops: list[SurfaceCircuitStep],
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
        p_prep: float = 0.0,
        add_detectors: bool = True,
    ) -> None:
        """Initialize Stim renderer.

        Args:
            p1: Single-qubit depolarizing error rate
            p2: Two-qubit depolarizing error rate
            p_meas: Measurement error rate
            p_prep: Initialization error rate
            add_detectors: Whether to add DETECTOR annotations
        """
        self.p1 = p1
        self.p2 = p2
        self.p_meas = p_meas
        self.p_prep = p_prep
        self.add_detectors = add_detectors

    def render(
        self,
        ops: list[SurfaceCircuitStep],
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
                if self.p_prep > 0:
                    lines.append(f"X_ERROR({self.p_prep}) {op.qubits[0]}")

            elif op.op_type == OpType.H:
                lines.append(f"H {op.qubits[0]}")
                if self.p1 > 0:
                    lines.append(f"DEPOLARIZE1({self.p1}) {op.qubits[0]}")

            elif op.op_type == OpType.SX:
                lines.append(f"SQRT_X {op.qubits[0]}")
                if self.p1 > 0:
                    lines.append(f"DEPOLARIZE1({self.p1}) {op.qubits[0]}")

            elif op.op_type == OpType.SXDG:
                lines.append(f"SQRT_X_DAG {op.qubits[0]}")
                if self.p1 > 0:
                    lines.append(f"DEPOLARIZE1({self.p1}) {op.qubits[0]}")

            elif op.op_type == OpType.SZ:
                lines.append(f"S {op.qubits[0]}")
                if self.p1 > 0:
                    lines.append(f"DEPOLARIZE1({self.p1}) {op.qubits[0]}")

            elif op.op_type == OpType.SZDG:
                lines.append(f"S_DAG {op.qubits[0]}")
                if self.p1 > 0:
                    lines.append(f"DEPOLARIZE1({self.p1}) {op.qubits[0]}")

            elif op.op_type == OpType.CX:
                c, t = op.qubits
                lines.append(f"CX {c} {t}")
                if self.p2 > 0:
                    lines.append(f"DEPOLARIZE2({self.p2}) {c} {t}")

            elif op.op_type == OpType.SZZ:
                a, b = op.qubits
                lines.append(f"SQRT_ZZ {a} {b}")
                if self.p2 > 0:
                    lines.append(f"DEPOLARIZE2({self.p2}) {a} {b}")

            elif op.op_type == OpType.SZZDG:
                a, b = op.qubits
                lines.append(f"SQRT_ZZ_DAG {a} {b}")
                if self.p2 > 0:
                    lines.append(f"DEPOLARIZE2({self.p2}) {a} {b}")

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

            elif op.op_type == OpType.TRACKED_PAULI:
                msg = (
                    "StimRenderer does not yet handle OpType.TRACKED_PAULI; "
                    "use TickCircuit / PauliFrameLookup path for twirled DEMs"
                )
                raise NotImplementedError(msg)

        # Add detector annotations if requested
        if self.add_detectors:
            lines.append("")
            lines.append("# Detectors")

            # Data prep fixes stabilizers matching the memory basis. The
            # complementary family is random but has an explicit init
            # measurement, which round 0 compares against.
            deterministic_type_round0 = "Z" if basis.upper() == "Z" else "X"
            init_baseline_type = "X" if basis.upper() == "Z" else "Z"

            # Syndrome detectors for X stabilizers
            for rnd in range(num_rounds):
                for s in geom.x_stabilizers:
                    curr_idx = stab_meas_record.get(("X", s.index, rnd))
                    if curr_idx is None:
                        continue
                    curr_offset = meas_count - curr_idx

                    if rnd == 0:
                        if init_baseline_type == "X":
                            init_idx = stab_meas_record[("X", s.index, -1)]
                            init_offset = meas_count - init_idx
                            lines.append(
                                f"DETECTOR({s.index}, 0, {rnd}) rec[{-curr_offset}] rec[{-init_offset}]",
                            )
                        elif deterministic_type_round0 == "X":
                            lines.append(
                                f"DETECTOR({s.index}, 0, {rnd}) rec[{-curr_offset}]",
                            )
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
                        if init_baseline_type == "Z":
                            init_idx = stab_meas_record[("Z", s.index, -1)]
                            init_offset = meas_count - init_idx
                            lines.append(
                                f"DETECTOR({det_x}, 1, {rnd}) rec[{-curr_offset}] rec[{-init_offset}]",
                            )
                        elif deterministic_type_round0 == "Z":
                            lines.append(
                                f"DETECTOR({det_x}, 1, {rnd}) rec[{-curr_offset}]",
                            )
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
                record_offsets = [*data_rec_offsets]
                if num_rounds > 0:
                    last_syn_idx = stab_meas_record[(stab_type, s.index, num_rounds - 1)]
                    record_offsets.append(meas_count - last_syn_idx)
                rec_str = " ".join(f"rec[{-off}]" for off in record_offsets)
                det_x = s.index if stab_type == "X" else num_x_anc + s.index
                det_y = 0 if stab_type == "X" else 1
                lines.append(
                    f"DETECTOR({det_x}, {det_y}, {num_rounds}) {rec_str}",
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
        _ops: list[SurfaceCircuitStep],
        _allocation: QubitAllocation,
        patch: SurfacePatch,
        _num_rounds: int,
        _basis: str,
        *,
        interaction_basis: str = "cx",
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
        return generate_guppy_source(patch, interaction_basis=interaction_basis)


class DagCircuitRenderer(CircuitRenderer):
    """Render circuit operations to PECOS DagCircuit."""

    def render(
        self,
        ops: list[SurfaceCircuitStep],
        _allocation: QubitAllocation,
        _patch: SurfacePatch,
        _num_rounds: int,
        _basis: str,
    ) -> DagCircuit:
        """Render to PECOS DagCircuit."""
        from pecos_rslib import DagCircuit, Gate, GateType

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

            elif op.op_type == OpType.SX:
                circuit.add_gate(Gate(GateType.SX, qubits=[op.qubits[0]]))

            elif op.op_type == OpType.SXDG:
                circuit.add_gate(Gate(GateType.SXdg, qubits=[op.qubits[0]]))

            elif op.op_type == OpType.SZ:
                circuit.sz([op.qubits[0]])

            elif op.op_type == OpType.SZDG:
                circuit.szdg([op.qubits[0]])

            elif op.op_type == OpType.X:
                circuit.x([op.qubits[0]])

            elif op.op_type == OpType.Z:
                circuit.z([op.qubits[0]])

            elif op.op_type == OpType.CX:
                circuit.cx([(op.qubits[0], op.qubits[1])])

            elif op.op_type == OpType.SZZ:
                circuit.szz([(op.qubits[0], op.qubits[1])])

            elif op.op_type == OpType.SZZDG:
                circuit.szzdg([(op.qubits[0], op.qubits[1])])

            elif op.op_type == OpType.MEASURE:
                q = op.qubits[0]
                if op.label.startswith(("sx", "sz")):
                    circuit.mz_free([q])
                    allocated.discard(q)
                else:
                    circuit.mz([q])

            elif op.op_type == OpType.TICK:
                pass  # DagCircuit doesn't have explicit ticks

            elif op.op_type == OpType.TRACKED_PAULI:
                msg = (
                    "DagCircuitRenderer does not yet handle OpType.TRACKED_PAULI; "
                    "use TickCircuit / PauliFrameLookup path for twirled DEMs"
                )
                raise NotImplementedError(msg)

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

    def __init__(
        self,
        *,
        add_detectors: bool = True,
        add_typed_annotations: bool = True,
    ) -> None:
        """Initialize TickCircuit renderer.

        Args:
            add_detectors: Whether to add detector/observable metadata.
            add_typed_annotations: Whether to also add typed Pauli annotations.
        """
        self.add_detectors = add_detectors
        self.add_typed_annotations = add_typed_annotations

    def render(
        self,
        ops: list[SurfaceCircuitStep],
        allocation: QubitAllocation,
        patch: SurfacePatch,
        num_rounds: int,
        basis: str,
    ) -> TickCircuit:
        """Render to PECOS TickCircuit.

        The tick structure follows Stim's pattern:
        - Tick: PZ data qubits
        - Tick: H for X-basis prep (if X-basis)
        - For each syndrome round:
            - Tick: PZ ancillas
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
        stab_meas_refs: dict[tuple[str, int, int], list] = {}
        final_meas_refs_by_qubit: dict[int, list] = {}
        current_round = -1
        current_phase = "prep"
        current_cx_round = 0
        final_meas_start = 0

        # Store tick-level metadata to apply at the end by tick index. Gate
        # metadata is attached immediately as each gate is emitted so it
        # participates in TickCircuit's batching decisions.
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

        # Helper to get stabilizer name for a two-qubit check interaction.
        def get_check_stabilizer(control: int, target: int, label: str = "") -> str:
            """Get stabilizer name for a two-qubit check gate (e.g., 'X0', 'Z2')."""
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

        def get_two_qubit_check_metadata(
            control: int,
            target: int,
            label: str = "",
            *,
            gate_kind: str,
        ) -> dict[str, object]:
            stab_label = get_check_stabilizer(control, target, label)
            if not stab_label:
                return {}
            metadata = get_stabilizer_metadata(stab_label)
            metadata["interaction_gate"] = gate_kind
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

        def is_syndrome_context(phase: str, round_index: int) -> bool:
            """Return whether the current context belongs to syndrome extraction."""
            if round_index >= 0 or phase.startswith("init_syndrome"):
                return True
            return round_index == -1 and (
                phase in {"syndrome_h_pre", "syndrome_h_post", "measure_ancilla"}
                or phase.startswith(("cx_round_", "szz_round_"))
            )

        def gate_metadata(meta: dict | None = None) -> dict:
            """Build metadata for the current gate context.

            Args:
                meta: Optional dict with gate metadata (e.g., {"label": "data[0]"})
            """
            context: dict[str, object] = {
                "phase": current_phase,
            }
            if is_syndrome_context(current_phase, current_round):
                context["syndrome_round"] = current_round
            if current_cx_round > 0:
                context["cx_round"] = current_cx_round
            if meta:
                return {**context, **meta}
            return context

        def apply_gate_metadata(handle: TickHandle, meta: dict | None = None) -> None:
            """Attach metadata to the gate most recently added to a handle."""
            handle.metas(gate_metadata(meta))

        def apply_measurement_metadata(meas_refs: list, meta: dict | None = None) -> None:
            """Attach metadata to the measurement gate just emitted."""
            if not meas_refs:
                return
            tick_idx, gate_idx, _ = meas_refs[0]
            for key, value in gate_metadata(meta).items():
                circuit.set_gate_meta(tick_idx, gate_idx, key, value)

        for op in ops:
            if op.op_type == OpType.COMMENT:
                # Track phase from comments
                if "syndrome_extraction round" in op.label:
                    current_round = int(op.label.split()[-1]) - 1
                    current_phase = "syndrome_prep"
                    current_cx_round = 0
                elif "init_" in op.label and "syndrome" in op.label:
                    current_round = -1
                    current_phase = "init_syndrome_prep"
                    current_cx_round = 0
                elif "Prepare ancillas" in op.label:
                    current_phase = "init_syndrome_prep" if current_round < 0 else "syndrome_prep"
                    current_cx_round = 0
                elif "Hadamard on X ancillas" in op.label or "Hadamard on SZZ ancillas" in op.label:
                    current_phase = (
                        "syndrome_h_pre"
                        if current_phase in {"syndrome_prep", "init_syndrome_prep"}
                        else "syndrome_h_post"
                    )
                elif "CX round" in op.label:
                    current_cx_round = int(op.label.split()[-1])
                    current_phase = f"cx_round_{current_cx_round}"
                elif "SZZ round" in op.label:
                    current_cx_round = int(op.label.split()[-1])
                    current_phase = f"szz_round_{current_cx_round}"
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
                    tick = tick.pz([q])
                mark_qubits_used([q])
                # Label helps identify which qubit (e.g., "data[0]", "ax0")
                meta = get_ancilla_gate_metadata(q, op.label)
                if op.label:
                    meta["label"] = op.label
                apply_gate_metadata(tick, meta or None)

            elif op.op_type == OpType.PREP:
                q = op.qubits[0]
                tick = get_tick_for_qubits([q]).pz([q])
                mark_qubits_used([q])
                meta = get_ancilla_gate_metadata(q, op.label)
                if op.label:
                    meta["label"] = op.label
                apply_gate_metadata(tick, meta or None)

            elif op.op_type == OpType.H:
                q = op.qubits[0]
                tick = get_tick_for_qubits([q]).h([q])
                mark_qubits_used([q])
                meta = get_ancilla_gate_metadata(q, op.label)
                if op.label:
                    meta["label"] = op.label
                apply_gate_metadata(tick, meta or None)

            elif op.op_type == OpType.F:
                q = op.qubits[0]
                tick = get_tick_for_qubits([q]).f([q])
                mark_qubits_used([q])
                meta = get_ancilla_gate_metadata(q, op.label)
                if op.label:
                    meta["label"] = op.label
                apply_gate_metadata(tick, meta or None)

            elif op.op_type == OpType.FDG:
                q = op.qubits[0]
                tick = get_tick_for_qubits([q]).fdg([q])
                mark_qubits_used([q])
                meta = get_ancilla_gate_metadata(q, op.label)
                if op.label:
                    meta["label"] = op.label
                apply_gate_metadata(tick, meta or None)

            elif op.op_type == OpType.SX:
                q = op.qubits[0]
                tick = get_tick_for_qubits([q]).sx([q])
                mark_qubits_used([q])
                meta = get_ancilla_gate_metadata(q, op.label)
                if op.label:
                    meta["label"] = op.label
                apply_gate_metadata(tick, meta or None)

            elif op.op_type == OpType.SXDG:
                q = op.qubits[0]
                tick = get_tick_for_qubits([q]).sxdg([q])
                mark_qubits_used([q])
                meta = get_ancilla_gate_metadata(q, op.label)
                if op.label:
                    meta["label"] = op.label
                apply_gate_metadata(tick, meta or None)

            elif op.op_type == OpType.SY:
                q = op.qubits[0]
                tick = get_tick_for_qubits([q]).sy([q])
                mark_qubits_used([q])
                meta = get_ancilla_gate_metadata(q, op.label)
                if op.label:
                    meta["label"] = op.label
                apply_gate_metadata(tick, meta or None)

            elif op.op_type == OpType.SYDG:
                q = op.qubits[0]
                tick = get_tick_for_qubits([q]).sydg([q])
                mark_qubits_used([q])
                meta = get_ancilla_gate_metadata(q, op.label)
                if op.label:
                    meta["label"] = op.label
                apply_gate_metadata(tick, meta or None)

            elif op.op_type == OpType.SZ:
                q = op.qubits[0]
                tick = get_tick_for_qubits([q]).sz([q])
                mark_qubits_used([q])
                meta = get_ancilla_gate_metadata(q, op.label)
                if op.label:
                    meta["label"] = op.label
                apply_gate_metadata(tick, meta or None)

            elif op.op_type == OpType.SZDG:
                q = op.qubits[0]
                tick = get_tick_for_qubits([q]).szdg([q])
                mark_qubits_used([q])
                meta = get_ancilla_gate_metadata(q, op.label)
                if op.label:
                    meta["label"] = op.label
                apply_gate_metadata(tick, meta or None)

            elif op.op_type == OpType.X:
                q = op.qubits[0]
                tick = get_tick_for_qubits([q]).x([q])
                mark_qubits_used([q])
                meta = get_ancilla_gate_metadata(q, op.label)
                if op.label:
                    meta["label"] = op.label
                apply_gate_metadata(tick, meta or None)

            elif op.op_type == OpType.Z:
                q = op.qubits[0]
                tick = get_tick_for_qubits([q]).z([q])
                mark_qubits_used([q])
                meta = get_ancilla_gate_metadata(q, op.label)
                if op.label:
                    meta["label"] = op.label
                if op.label.startswith("szz_virtual_prefix:"):
                    meta[PHYSICAL_DURATION_META_KEY] = 0.0
                apply_gate_metadata(tick, meta or None)

            elif op.op_type == OpType.CX:
                qubits = op.qubits
                tick = get_tick_for_qubits(qubits).cx([(qubits[0], qubits[1])])
                mark_qubits_used(qubits)
                meta = get_two_qubit_check_metadata(qubits[0], qubits[1], op.label, gate_kind="CX")
                if op.label:
                    meta["label"] = op.label
                apply_gate_metadata(tick, meta or None)

            elif op.op_type == OpType.SZZ:
                qubits = op.qubits
                tick = get_tick_for_qubits(qubits).szz([(qubits[0], qubits[1])])
                mark_qubits_used(qubits)
                meta = get_two_qubit_check_metadata(qubits[0], qubits[1], op.label, gate_kind="SZZ")
                if op.label:
                    meta["label"] = op.label
                apply_gate_metadata(tick, meta or None)

            elif op.op_type == OpType.SZZDG:
                qubits = op.qubits
                tick = get_tick_for_qubits(qubits).szzdg([(qubits[0], qubits[1])])
                mark_qubits_used(qubits)
                meta = get_two_qubit_check_metadata(qubits[0], qubits[1], op.label, gate_kind="SZZdg")
                if op.label:
                    meta["label"] = op.label
                apply_gate_metadata(tick, meta or None)

            elif op.op_type == OpType.MEASURE:
                q = op.qubits[0]
                if op.label.startswith(("sx", "sz")):
                    meas_refs = get_tick_for_qubits([q]).mz_free([q])
                    allocated.discard(q)
                else:
                    meas_refs = get_tick_for_qubits([q]).mz([q])
                mark_qubits_used([q])
                # Label helps identify measurement (e.g., "sx0", "sz0", "final[0]")
                meta = get_ancilla_gate_metadata(q, op.label)
                if op.label:
                    meta["label"] = op.label
                apply_measurement_metadata(meas_refs, meta or None)

                # Track measurement index and refs for detectors
                if op.label.startswith("sx"):
                    stab_idx = int(op.label[2:])
                    stab_meas_record[("X", stab_idx, current_round)] = meas_count
                    stab_meas_refs[("X", stab_idx, current_round)] = meas_refs
                elif op.label.startswith("sz"):
                    stab_idx = int(op.label[2:])
                    stab_meas_record[("Z", stab_idx, current_round)] = meas_count
                    stab_meas_refs[("Z", stab_idx, current_round)] = meas_refs
                elif op.label.startswith("final"):
                    if "final[0]" in op.label:
                        final_meas_start = meas_count
                    # Track all final measurement refs by data qubit
                    final_meas_refs_by_qubit[q] = meas_refs
                meas_count += 1

            elif op.op_type == OpType.TICK:
                current_tick_handle = None
                qubits_in_current_tick = set()

            elif op.op_type == OpType.TRACKED_PAULI:
                from pecos_rslib import PauliString

                q = op.qubits[0]
                kind, sep, site_suffix = op.label.partition("@")
                pauli_ctor = {
                    "X": PauliString.X,
                    "Y": PauliString.Y,
                    "Z": PauliString.Z,
                }.get(kind)
                if pauli_ctor is None or sep != "@" or not site_suffix.startswith(("s", "g")):
                    msg = (
                        "OpType.TRACKED_PAULI requires label of the form "
                        f"'{{X|Y|Z}}@s<site_idx>' or "
                        f"'{{X|Y|Z}}@g<site_idx>o<operand_idx>', got {op.label!r}"
                    )
                    raise ValueError(msg)
                circuit.tracked_pauli(
                    pauli_ctor(q),
                    label=f"twirl_{site_suffix}_q{q}_{kind}",
                )

        # Apply tick-level metadata in place. Gate metadata is attached as each
        # gate is emitted so batching decisions can account for it immediately.
        for tick_idx, tick_meta in all_tick_metadata.items():
            # Set tick-level metadata
            circuit.set_tick_meta(tick_idx, "phase", tick_meta["phase"])
            if is_syndrome_context(str(tick_meta["phase"]), int(tick_meta["round"])):
                circuit.set_tick_meta(tick_idx, "syndrome_round", tick_meta["round"])
            if tick_meta["cx_round"] > 0:
                circuit.set_tick_meta(tick_idx, "cx_round", tick_meta["cx_round"])

        # Add detector annotations as metadata
        if self.add_detectors:
            geom = patch.geometry
            num_x_anc = len(geom.x_stabilizers)
            deterministic_type_round0 = "Z" if basis.upper() == "Z" else "X"
            init_baseline_type = "X" if basis.upper() == "Z" else "Z"

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
                        if init_baseline_type == "X":
                            init_idx = stab_meas_record[("X", s.index, -1)]
                            init_offset = meas_count - init_idx
                            detectors.append(
                                {
                                    "id": detector_id,
                                    "coords": [s.index, 0, rnd],
                                    "records": [-curr_offset, -init_offset],
                                },
                            )
                            detector_id += 1
                        elif deterministic_type_round0 == "X":
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
                        if init_baseline_type == "Z":
                            init_idx = stab_meas_record[("Z", s.index, -1)]
                            init_offset = meas_count - init_idx
                            detectors.append(
                                {
                                    "id": detector_id,
                                    "coords": [det_x, 1, rnd],
                                    "records": [-curr_offset, -init_offset],
                                },
                            )
                            detector_id += 1
                        elif deterministic_type_round0 == "Z":
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
                records = [*data_rec_offsets]
                if num_rounds > 0:
                    last_syn_idx = stab_meas_record[(stab_type, s.index, num_rounds - 1)]
                    records.append(-(meas_count - last_syn_idx))
                det_x = s.index if stab_type == "X" else num_x_anc + s.index
                det_y = 0 if stab_type == "X" else 1
                detectors.append(
                    {
                        "id": detector_id,
                        "coords": [det_x, det_y, num_rounds],
                        "records": records,
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

            # Store as metadata (legacy path for DemBuilder/caching)
            circuit.set_meta("detectors", json.dumps(detectors))
            circuit.set_meta("observables", json.dumps(observables))
            circuit.set_meta("num_measurements", str(meas_count))
            circuit.set_meta("num_detectors", str(len(detectors)))

            # Also add typed PauliAnnotation annotations (new path) when the
            # caller wants direct Pauli annotations in addition to legacy
            # measurement-record metadata. Native surface DEM construction uses
            # the JSON metadata below and disables these annotations to avoid
            # mixing two independent observable sources in the same influence
            # map.
            if self.add_typed_annotations:
                self._add_typed_annotations(
                    circuit,
                    geom,
                    num_rounds,
                    basis,
                    stab_meas_refs,
                    final_meas_refs_by_qubit,
                    deterministic_type_round0,
                    init_baseline_type,
                )
        circuit.set_meta("basis", basis.upper())
        circuit.set_meta("ancilla_budget", str(allocation.total - len(allocation.data_qubits)))

        return circuit

    @staticmethod
    def _add_typed_annotations(
        circuit: TickCircuit,
        geom: object,
        num_rounds: int,
        basis: str,
        stab_meas_refs: dict,
        final_meas_refs_by_qubit: dict,
        deterministic_type_round0: str,
        init_baseline_type: str,
    ) -> None:
        """Add typed PauliAnnotation detectors and observables to the circuit.

        This mirrors the JSON detector logic but uses the new annotation API
        with TickMeasRef measurement references.
        """
        # Syndrome detectors for X stabilizers
        for rnd in range(num_rounds):
            for s in geom.x_stabilizers:
                curr_refs = stab_meas_refs.get(("X", s.index, rnd))
                if curr_refs is None:
                    continue
                if rnd == 0:
                    if init_baseline_type == "X":
                        init_refs = stab_meas_refs.get(("X", s.index, -1), [])
                        circuit.detector(init_refs + curr_refs, label=f"Sx{s.index}_r{rnd}")
                    elif deterministic_type_round0 == "X":
                        circuit.detector(curr_refs, label=f"Sx{s.index}_r{rnd}")
                else:
                    prev_refs = stab_meas_refs.get(("X", s.index, rnd - 1), [])
                    circuit.detector(prev_refs + curr_refs, label=f"Sx{s.index}_r{rnd}")

        # Syndrome detectors for Z stabilizers
        for rnd in range(num_rounds):
            for s in geom.z_stabilizers:
                curr_refs = stab_meas_refs.get(("Z", s.index, rnd))
                if curr_refs is None:
                    continue
                if rnd == 0:
                    if init_baseline_type == "Z":
                        init_refs = stab_meas_refs.get(("Z", s.index, -1), [])
                        circuit.detector(init_refs + curr_refs, label=f"Sz{s.index}_r{rnd}")
                    elif deterministic_type_round0 == "Z":
                        circuit.detector(curr_refs, label=f"Sz{s.index}_r{rnd}")
                else:
                    prev_refs = stab_meas_refs.get(("Z", s.index, rnd - 1), [])
                    circuit.detector(prev_refs + curr_refs, label=f"Sz{s.index}_r{rnd}")

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
            # Data qubit measurement refs for this stabilizer
            data_refs = []
            for dq in s.data_qubits:
                if dq in final_meas_refs_by_qubit:
                    data_refs.extend(final_meas_refs_by_qubit[dq])
            # Last syndrome round ref
            last_syn_refs = stab_meas_refs.get(
                (stab_type, s.index, num_rounds - 1),
                [],
            )
            label_prefix = "Sx" if stab_type == "X" else "Sz"
            circuit.detector(
                data_refs + last_syn_refs,
                label=f"{label_prefix}{s.index}_final",
            )

        # Logical observable
        obs_refs = []
        for q in logical_qubits:
            if q in final_meas_refs_by_qubit:
                obs_refs.extend(final_meas_refs_by_qubit[q])
        if obs_refs:
            circuit.observable(obs_refs, label=f"logical_{basis.upper()}")


# Convenience functions


def generate_stim_from_patch(
    patch: SurfacePatch,
    num_rounds: int,
    basis: str = "Z",
    *,
    ancilla_budget: int | None = None,
    interaction_basis: str | None = None,
    check_plan: str | None = None,
    p1: float = 0.0,
    p2: float = 0.0,
    p_meas: float = 0.0,
    p_prep: float = 0.0,
    add_detectors: bool = True,
) -> str:
    """Generate Stim circuit from SurfacePatch.

    Args:
        patch: Surface code patch
        num_rounds: Number of syndrome rounds
        basis: 'Z' or 'X'
        ancilla_budget: Optional cap on simultaneously live ancillas
        interaction_basis: Surface-memory two-qubit interaction basis.
        check_plan: Named surface check-plan preset.
        p1: Single-qubit error rate
        p2: Two-qubit error rate
        p_meas: Measurement error rate
        p_prep: Initialization error rate
        add_detectors: Whether to add detector/observable annotations.

    Returns:
        Stim circuit string
    """
    ops, allocation = build_surface_code_circuit(
        patch,
        num_rounds,
        basis,
        ancilla_budget,
        interaction_basis=interaction_basis,
        check_plan=check_plan,
    )
    renderer = StimRenderer(p1=p1, p2=p2, p_meas=p_meas, p_prep=p_prep, add_detectors=add_detectors)
    return renderer.render(ops, allocation, patch, num_rounds, basis)


def generate_guppy_from_patch(
    patch: SurfacePatch,
    _num_rounds: int = 1,
    _basis: str = "Z",
    *,
    interaction_basis: str | None = None,
    check_plan: str | None = None,
    clifford_frame_policy: str | None = None,
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
        interaction_basis: Surface-memory two-qubit interaction basis.
        check_plan: Named surface check-plan preset.
        clifford_frame_policy: Optional source-level Clifford-deformation
            policy for SZZ/SZZdg surface-code generation.

    Returns:
        Guppy source code string (full module)
    """
    from pecos.guppy.surface import generate_guppy_source

    return generate_guppy_source(
        patch,
        interaction_basis=interaction_basis,
        check_plan=check_plan,
        clifford_frame_policy=clifford_frame_policy,
    )


def generate_dag_circuit_from_patch(
    patch: SurfacePatch,
    num_rounds: int,
    basis: str = "Z",
    ancilla_budget: int | None = None,
    *,
    interaction_basis: str | None = None,
    check_plan: str | None = None,
) -> DagCircuit:
    """Generate PECOS DagCircuit from SurfacePatch.

    Args:
        patch: Surface code patch
        num_rounds: Number of syndrome rounds
        basis: 'Z' or 'X'
        ancilla_budget: Optional cap on simultaneously live ancillas
        interaction_basis: Surface-memory two-qubit interaction basis.
        check_plan: Named surface check-plan preset.

    Returns:
        PECOS DagCircuit instance
    """
    ops, allocation = build_surface_code_circuit(
        patch,
        num_rounds,
        basis,
        ancilla_budget,
        interaction_basis=interaction_basis,
        check_plan=check_plan,
    )
    renderer = DagCircuitRenderer()
    return renderer.render(ops, allocation, patch, num_rounds, basis)


def generate_tick_circuit_from_patch(
    patch: SurfacePatch,
    num_rounds: int,
    basis: str = "Z",
    *,
    add_detectors: bool = True,
    add_typed_annotations: bool = True,
    ancilla_budget: int | None = None,
    twirl: TwirlConfig | None = None,
    interaction_basis: str | None = None,
    check_plan: str | None = None,
    szz_physical_prefixes: bool = False,
    clifford_frame_policy: str | None = None,
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
        add_detectors: Whether to add detector/observable metadata.
        add_typed_annotations: Whether to also add typed Pauli annotations.
        ancilla_budget: Optional cap on simultaneously live ancillas
        twirl: Optional Pauli-frame randomization layout. When supplied,
            tracked-Pauli annotations are emitted even if
            ``add_typed_annotations`` is false; that flag controls detector
            and observable typed annotations, not the twirl lookup channel.
        interaction_basis: Surface-memory two-qubit interaction basis.
        check_plan: Named surface check-plan preset.
        szz_physical_prefixes: If true, lower the abstract SZZ single-qubit
            scaffold into physical prefix pulses for native DEM analysis.
        clifford_frame_policy: Optional source-level Clifford-deformation
            policy for SZZ generation. Currently supports global uniform-axis
            frames.

    Returns:
        PECOS TickCircuit instance
    """
    resolved_plan = resolve_surface_check_plan(
        interaction_basis=interaction_basis,
        check_plan=check_plan,
    )
    require_current_surface_check_plan_renderer(
        resolved_plan,
        context="abstract surface TickCircuit generation",
    )
    interaction_basis = _normalize_interaction_basis(resolved_plan.interaction_basis)
    if szz_physical_prefixes and interaction_basis != "szz":
        msg = "szz_physical_prefixes=True requires interaction_basis='szz'"
        raise ValueError(msg)
    ops, allocation = build_surface_code_circuit(
        patch,
        num_rounds,
        basis,
        ancilla_budget,
        twirl=twirl,
        interaction_basis=interaction_basis,
        check_plan=resolved_plan.plan_id,
        clifford_frame_policy=clifford_frame_policy,
    )
    if szz_physical_prefixes:
        ops = _lower_szz_forward_flow_ops(ops)
    renderer = TickCircuitRenderer(
        add_detectors=add_detectors,
        add_typed_annotations=add_typed_annotations,
    )
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
    p1_gate_rates: Mapping[str, float] | None = None,
    p2: float = 0.0,
    p_meas: float = 0.0,
    p_prep: float = 0.0,
) -> str:
    """Convert TickCircuit to Stim circuit string.

    This makes TickCircuit the source of truth for circuit structure,
    with Stim circuit being derived from it for DEM generation.

    Args:
        tc: TickCircuit instance with detector/observable metadata
        p1: Single-qubit error rate
        p1_gate_rates: Optional per-gate override for single-qubit error
            rates. Gate names are PECOS ``GateType`` names such as ``"Z"``,
            ``"SZ"``, and ``"SZdg"``. The surface SZZ reference path uses
            this to mirror the staged PECOS device model where Z/SZ/SZdg frame
            updates are virtual and p1-free.
        p2: Two-qubit error rate
        p_meas: Measurement error rate
        p_prep: Initialization error rate

    Returns:
        Stim circuit string
    """
    import json
    import math

    lines = []

    simple_gate_map = {
        "H": ("H", "single"),
        "SX": ("SQRT_X", "single"),
        "SXdg": ("SQRT_X_DAG", "single"),
        "SY": ("SQRT_Y", "single"),
        "SYdg": ("SQRT_Y_DAG", "single"),
        "SZ": ("S", "single"),
        "SZdg": ("S_DAG", "single"),
        "X": ("X", "single"),
        "Y": ("Y", "single"),
        "Z": ("Z", "single"),
        "CX": ("CX", "two"),
        "CY": ("CY", "two"),
        "CZ": ("CZ", "two"),
        "SZZ": ("SQRT_ZZ", "two"),
        "SZZdg": ("SQRT_ZZ_DAG", "two"),
        "MZ": ("M", "measure"),
        "MeasureFree": ("M", "measure"),
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

        if gate_name == "F":
            return [("S_DAG", qubits), ("H", qubits)], "single"

        if gate_name == "Fdg":
            return [("H", qubits), ("S", qubits)], "single"

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
        for gate in tick.gate_batches():
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

            p1_for_gate = p1 if p1_gate_rates is None else float(p1_gate_rates.get(gate.gate_type.name, p1))
            if noise_kind == "single" and p1_for_gate > 0:
                lines.append(f"DEPOLARIZE1({p1_for_gate}) {qubit_str}")
            elif noise_kind == "two" and p2 > 0:
                lines.append(f"DEPOLARIZE2({p2}) {qubit_str}")
            elif noise_kind == "prep" and p_prep > 0:
                lines.append(f"X_ERROR({p_prep}) {qubit_str}")

        # Add TICK after each tick (except the last)
        if tick_idx < tc.num_ticks() - 1:
            lines.append("TICK")

    # Add DETECTOR annotations from TickCircuit metadata
    detectors_json = tc.get_meta("detectors")
    if detectors_json:
        detectors = json.loads(detectors_json)
        num_measurements = int(tc.get_meta("num_measurements") or "0")
        for det in detectors:
            coords = det["coords"]
            records = _metadata_record_offsets(det, num_measurements)
            coord_str = ", ".join(str(c) for c in coords)
            record_str = " ".join(f"rec[{r}]" for r in records)
            lines.append(f"DETECTOR({coord_str}) {record_str}")

    # Add OBSERVABLE_INCLUDE from metadata
    observables_json = tc.get_meta("observables")
    if observables_json:
        observables = json.loads(observables_json)
        num_measurements = int(tc.get_meta("num_measurements") or "0")
        for obs in observables:
            obs_id = obs["id"]
            records = _metadata_record_offsets(obs, num_measurements)
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
        p_prep=p,
    )
    circuit = stim.Circuit(circuit_str)
    return str(circuit.detector_error_model())


def generate_dem_from_tick_circuit_via_pauli_frame(
    tc: TickCircuit,
    *,
    p1: float = 0.01,
    p2: float = 0.01,
    p2_weights: Mapping[str, float] | None = None,
    p_meas: float = 0.01,
    p_prep: float = 0.01,
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
        p2_weights: Optional relative probabilities over the 15 non-identity
            two-qubit Pauli errors (``IX`` through ``ZZ``). Values must sum to
            1.0; ``p2`` remains the total two-qubit error rate.
        p_meas: Measurement error rate
        p_prep: Initialization (prep) error rate

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
        for rec in _metadata_record_offsets(det, num_measurements):
            abs_meas = num_measurements + rec  # rec is negative
            meas_to_detectors[abs_meas].append(det_id)

    meas_to_observables: dict[int, list[int]] = defaultdict(list)
    for obs in observables:
        obs_id = obs["id"]
        for rec in _metadata_record_offsets(obs, num_measurements):
            abs_meas = num_measurements + rec
            meas_to_observables[abs_meas].append(obs_id)

    # Build circuit structure for simulation
    # We need: list of (tick_idx, gate_type, qubits, meas_idx_if_applicable)
    circuit_ops: list[tuple[int, str, list[int], int | None]] = []
    meas_counter = 0

    for tick_idx in range(tc.num_ticks()):
        tick = tc.get_tick(tick_idx)
        for gate in tick.gate_batches():
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
    two_pauli_labels = tuple(
        f"{p_ctrl}{p_targ}"
        for p_ctrl in ("I", "X", "Y", "Z")
        for p_targ in ("I", "X", "Y", "Z")
        if not (p_ctrl == "I" and p_targ == "I")
    )
    if p2_weights is None:
        two_paulis = tuple((label[0], label[1], 1.0 / 15.0) for label in two_pauli_labels)
    else:
        from math import isfinite

        weights = {str(label).upper(): float(weight) for label, weight in p2_weights.items()}
        unknown_labels = sorted(set(weights) - set(two_pauli_labels))
        if unknown_labels:
            message = f"p2_weights contains invalid Pauli labels: {unknown_labels}"
            raise ValueError(message)
        if any(not isfinite(weight) or weight < 0.0 for weight in weights.values()):
            message = "p2_weights values must be finite and non-negative"
            raise ValueError(message)
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) >= 1.0e-6:
            message = f"p2_weights relative probabilities must sum to 1.0, got {weight_sum}"
            raise ValueError(message)
        two_paulis = tuple((label[0], label[1], weight) for label, weight in sorted(weights.items()) if weight > 0.0)

    # Process each gate as a potential error location
    for op_idx, (_tick_idx, gate_name, qubits, meas_idx) in enumerate(circuit_ops):
        if gate_name in ("QAlloc", "PZ") and p_prep > 0:
            # Initialization error: X error after prep
            q = qubits[0]
            dets, obs = simulate_error(op_idx + 1, {q: "X"})
            if dets or obs:
                key = (frozenset(dets), frozenset(obs))
                error_mechanisms[key] += p_prep

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
            for p_ctrl, p_targ, relative_probability in two_paulis:
                frame = {}
                if p_ctrl != "I":
                    frame[ctrl] = p_ctrl
                if p_targ != "I":
                    frame[targ] = p_targ
                dets, obs = simulate_error(op_idx + 1, frame)
                if dets or obs:
                    key = (frozenset(dets), frozenset(obs))
                    error_mechanisms[key] += p2 * relative_probability

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
    p1_gate_rates: Mapping[str, float] | None = None,
    p2: float = 0.01,
    p_meas: float = 0.01,
    p_prep: float = 0.01,
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
        p1_gate_rates: Optional per-gate override for single-qubit
            depolarizing rates. Gate names are PECOS ``GateType`` names. The
            surface SZZ reference path uses this to mirror the staged PECOS
            device model where Z/SZ/SZdg frame updates are virtual and p1-free.
        p2: Two-qubit depolarizing error rate
        p_meas: Measurement error rate
        p_prep: Initialization (prep) error rate
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

    stim_str = tick_circuit_to_stim(
        tc,
        p1=p1,
        p1_gate_rates=p1_gate_rates,
        p2=p2,
        p_meas=p_meas,
        p_prep=p_prep,
    )
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
        gates = tick.gate_batches()
        for gate in gates:
            gate_type = str(gate.gate_type)
            if "MZ" in gate_type or "MeasureFree" in gate_type:
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


def _build_canonical_dem_influence_map(
    dag: DagCircuit,
    *,
    include_circuit_annotations: bool = False,
) -> object:
    """Build the influence map used by the metadata-driven Rust DEM builder."""
    from pecos.qec import DagFaultAnalyzer

    analyzer = DagFaultAnalyzer(dag)
    influence_map = analyzer.build_influence_map()
    if include_circuit_annotations:
        from pecos.qec import InfluenceBuilder

        annotation_builder = InfluenceBuilder(dag)
        annotation_builder.with_circuit_annotations()
        annotation_map = annotation_builder.build()
        merge_dem_outputs = getattr(influence_map, "merge_dem_outputs_from", None)
        if merge_dem_outputs is not None:
            merge_dem_outputs(annotation_map)
    return influence_map


def _metadata_uses_record_offsets(*metadata_jsons: str | None) -> bool:
    """Return whether detector/observable metadata uses positional records."""
    import json

    for metadata_json in metadata_jsons:
        if not metadata_json:
            continue
        for entry in json.loads(metadata_json):
            if entry.get("records"):
                return True
    return False


def _metadata_record_offsets(entry: dict[str, object], num_measurements: int) -> list[int]:
    """Return Stim-style negative record offsets for a metadata entry."""
    records = entry.get("records")
    if records is not None:
        return [int(record) for record in records]  # type: ignore[union-attr]

    meas_ids = entry.get("meas_ids")
    if meas_ids is not None:
        return [int(meas_id) - num_measurements for meas_id in meas_ids]  # type: ignore[union-attr]

    msg = "detector/observable metadata entry must define either 'records' or 'meas_ids'"
    raise ValueError(msg)


def generate_dem_from_tick_circuit(
    tc: TickCircuit,
    *,
    p1: float = 0.01,
    p1_weights: Mapping[str, float] | None = None,
    p2: float = 0.01,
    p2_weights: Mapping[str, float] | None = None,
    p_meas: float = 0.01,
    p_prep: float = 0.01,
    p_idle: float | None = None,
    t1: float | None = None,
    t2: float | None = None,
    p_idle_linear_rate: float | None = None,
    p_idle_quadratic_rate: float | None = None,
    p_idle_x_linear_rate: float | None = None,
    p_idle_y_linear_rate: float | None = None,
    p_idle_z_linear_rate: float | None = None,
    p_idle_x_quadratic_rate: float | None = None,
    p_idle_y_quadratic_rate: float | None = None,
    p_idle_z_quadratic_rate: float | None = None,
    p_idle_quadratic_sine_rate: float | None = None,
    p_idle_x_quadratic_sine_rate: float | None = None,
    p_idle_y_quadratic_sine_rate: float | None = None,
    p_idle_z_quadratic_sine_rate: float | None = None,
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
        p1: Single-qubit Pauli error rate
        p1_weights: Optional relative probabilities over single-qubit Pauli
            errors (``X``, ``Y``, ``Z``). Values must sum to 1.0; ``p1``
            remains the total single-qubit error rate.
        p2: Two-qubit depolarizing error rate
        p2_weights: Optional relative probabilities over the 15 non-identity
            two-qubit Pauli errors (``IX`` through ``ZZ``). Values must sum to
            1.0; ``p2`` remains the total two-qubit error rate.
        p_meas: Measurement error rate
        p_prep: Initialization (prep) error rate
        p_idle: Optional idle noise rate per explicit idle-gate time unit.
            The caller is responsible for inserting idle gates where needed.
        t1: Optional T1 relaxation time for explicit idle gates.
        t2: Optional T2 dephasing time for explicit idle gates.
        p_idle_linear_rate: Optional legacy alias for stochastic Z-memory rate
            linear in idle duration.
        p_idle_quadratic_rate: Optional legacy alias for stochastic Z-memory rate
            quadratic in idle duration.
        p_idle_x_linear_rate: Optional stochastic X-memory rate linear in idle duration.
        p_idle_y_linear_rate: Optional stochastic Y-memory rate linear in idle duration.
        p_idle_z_linear_rate: Optional stochastic Z-memory rate linear in idle duration.
        p_idle_x_quadratic_rate: Optional stochastic X-memory rate quadratic in idle duration.
        p_idle_y_quadratic_rate: Optional stochastic Y-memory rate quadratic in idle duration.
        p_idle_z_quadratic_rate: Optional stochastic Z-memory rate quadratic in idle duration.
        p_idle_quadratic_sine_rate: Optional legacy alias for stochastic Z-memory
            rate with probability ``sin(rate * duration)^2``.
        p_idle_x_quadratic_sine_rate: Optional stochastic X-memory sine-law rate.
        p_idle_y_quadratic_sine_rate: Optional stochastic Y-memory sine-law rate.
        p_idle_z_quadratic_sine_rate: Optional stochastic Z-memory sine-law rate.
        decompose_errors: If True (default), decompose hyperedge errors into
            graphlike components using the `^` separator. Set to False to
            output raw hyperedges. Ignored if maximal_decomposition=True.
        maximal_decomposition: If True, maximally decompose all mechanisms
            into single-detector components. This produces output similar
            to other tools that prefer maximal decomposition.

    Returns:
        DEM string in Stim-compatible format
    """
    from pecos.qec import DemBuilder

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
    metadata_uses_records = _metadata_uses_record_offsets(detectors_json, observables_json)

    # Convert TickCircuit to DagCircuit and build influence map
    dag = tc.to_dag_circuit()
    influence_map = _build_canonical_dem_influence_map(dag)

    # Build DEM using Rust DemBuilder
    builder = DemBuilder(influence_map)
    builder.with_noise(
        p1,
        p2,
        p_meas,
        p_prep,
        p1_weights=p1_weights,
        p2_weights=p2_weights,
        p_idle=p_idle,
        t1=t1,
        t2=t2,
        p_idle_linear_rate=p_idle_linear_rate,
        p_idle_quadratic_rate=p_idle_quadratic_rate,
        p_idle_x_linear_rate=p_idle_x_linear_rate,
        p_idle_y_linear_rate=p_idle_y_linear_rate,
        p_idle_z_linear_rate=p_idle_z_linear_rate,
        p_idle_x_quadratic_rate=p_idle_x_quadratic_rate,
        p_idle_y_quadratic_rate=p_idle_y_quadratic_rate,
        p_idle_z_quadratic_rate=p_idle_z_quadratic_rate,
        p_idle_quadratic_sine_rate=p_idle_quadratic_sine_rate,
        p_idle_x_quadratic_sine_rate=p_idle_x_quadratic_sine_rate,
        p_idle_y_quadratic_sine_rate=p_idle_y_quadratic_sine_rate,
        p_idle_z_quadratic_sine_rate=p_idle_z_quadratic_sine_rate,
    )
    if hasattr(builder, "with_exact_branch_replay_circuit"):
        builder = builder.with_exact_branch_replay_circuit(dag)
    builder.with_num_measurements(num_measurements)
    if metadata_uses_records:
        builder.with_measurement_order(measurement_order)
    builder.with_detectors_json(detectors_json)
    if observables_json:
        builder.with_observables_json(observables_json)

    dem = builder.build_with_source_tracking()

    if maximal_decomposition:
        return _maximally_decompose_graphlike_dem(dem.to_string_decomposed())
    if decompose_errors:
        source_graphlike = getattr(dem, "to_string_source_graphlike_decomposed", None)
        if source_graphlike is not None:
            return source_graphlike()
        return dem.to_string_decomposed()
    return dem.to_string()


def generate_dem_from_tick_circuit_via_autodetection(
    tc: TickCircuit,
    *,
    tracked_z_qubits: list[int] | None = None,
    tracked_x_qubits: list[int] | None = None,
    p1: float = 0.01,
    p2: float = 0.01,
    p_meas: float = 0.01,
    p_prep: float = 0.01,
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
        tracked_z_qubits: Qubit indices for a tracked Z Pauli (for X error tracking)
        tracked_x_qubits: Qubit indices for a tracked X Pauli (for Z error tracking)
        p1: Single-qubit depolarizing error rate
        p2: Two-qubit depolarizing error rate
        p_meas: Measurement error rate
        p_prep: Initialization (prep) error rate

    Returns:
        PECOS DEM string. With no tracked Paulis this is Stim-compatible;
        tracked Paulis are represented with PECOS `pecos_tracked_pauli`
        metadata lines.
    """
    import json
    from collections import defaultdict

    from pecos.qec import PAULI_X, PAULI_Y, PAULI_Z, InfluenceBuilder

    # Convert TickCircuit to DagCircuit
    dag = tc.to_dag_circuit()

    # Build influence map with auto-discovered detectors
    builder = InfluenceBuilder(dag)
    if tracked_x_qubits:
        builder.with_tracked_x(tracked_x_qubits)
    if tracked_z_qubits:
        builder.with_tracked_z(tracked_z_qubits)
    influence_map = builder.build()

    # Get all fault locations and auto-discovered detectors
    locations = influence_map.get_locations()
    num_detectors = influence_map.num_detectors

    # Collect error mechanisms: (detectors, DEM outputs) -> probability
    error_mechanisms: dict[tuple[frozenset[int], frozenset[int]], float] = defaultdict(
        float,
    )

    # Process each fault location
    for loc_idx, loc in enumerate(locations):
        gate_type = loc.gate_type

        if "PZ" in gate_type or "QAlloc" in gate_type:
            if p_prep <= 0:
                continue
            for pauli in [PAULI_X]:
                dets = set(influence_map.get_detector_indices(loc_idx, pauli))
                dem_outputs = set(influence_map.get_observable_indices(loc_idx, pauli))
                if dets or dem_outputs:
                    key = (frozenset(dets), frozenset(dem_outputs))
                    error_mechanisms[key] += p_prep

        elif "MZ" in gate_type:
            if p_meas <= 0:
                continue
            for pauli in [PAULI_X]:
                dets = set(influence_map.get_detector_indices(loc_idx, pauli))
                dem_outputs = set(influence_map.get_observable_indices(loc_idx, pauli))
                if dets or dem_outputs:
                    key = (frozenset(dets), frozenset(dem_outputs))
                    error_mechanisms[key] += p_meas

        elif "CX" in gate_type:
            if p2 <= 0:
                continue
            for pauli in [PAULI_X, PAULI_Y, PAULI_Z]:
                dets = set(influence_map.get_detector_indices(loc_idx, pauli))
                dem_outputs = set(influence_map.get_observable_indices(loc_idx, pauli))
                if dets or dem_outputs:
                    key = (frozenset(dets), frozenset(dem_outputs))
                    error_mechanisms[key] += p2 / 3

        elif "H" in gate_type:
            if p1 <= 0:
                continue
            for pauli in [PAULI_X, PAULI_Y, PAULI_Z]:
                dets = set(influence_map.get_detector_indices(loc_idx, pauli))
                dem_outputs = set(influence_map.get_observable_indices(loc_idx, pauli))
                if dets or dem_outputs:
                    key = (frozenset(dets), frozenset(dem_outputs))
                    error_mechanisms[key] += p1 / 3

    # Generate DEM output
    # Add detector declarations (auto-discovered, no coordinates)
    lines = [f"detector D{det_idx}" for det_idx in range(num_detectors)]

    def _pauli_string(pauli: str, qubits: list[int] | None) -> str:
        if not qubits:
            return "+I"
        return "+" + " ".join(f"{pauli}{q}" for q in qubits)

    tracked_pauli_metadata = []
    if tracked_x_qubits:
        tracked_pauli_metadata.append(
            {
                "id": len(tracked_pauli_metadata),
                "kind": "tracked_pauli",
                "label": "tracked_x",
                "pauli": _pauli_string("X", tracked_x_qubits),
            },
        )
    if tracked_z_qubits:
        tracked_pauli_metadata.append(
            {
                "id": len(tracked_pauli_metadata),
                "kind": "tracked_pauli",
                "label": "tracked_z",
                "pauli": _pauli_string("Z", tracked_z_qubits),
            },
        )
    lines.extend(
        f"pecos_tracked_pauli {json.dumps(metadata, separators=(',', ':'))}" for metadata in tracked_pauli_metadata
    )

    # Add error mechanisms
    for (dets, dem_outputs), prob in sorted(
        error_mechanisms.items(),
        key=lambda x: (sorted(x[0][0]), sorted(x[0][1])),
    ):
        if prob > 0 and (dets or dem_outputs):
            det_str = " ".join(f"D{d}" for d in sorted(dets))
            dem_output_str = " ".join(f"L{idx}" for idx in sorted(dem_outputs))
            targets = f"{det_str} {dem_output_str}".strip()
            lines.append(f"error({prob:.6g}) {targets}")

    return "\n".join(lines)
