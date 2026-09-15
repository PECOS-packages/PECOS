# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Logical circuit builder for surface codes with transversal gates.

Generates PECOS TickCircuit circuits natively, with Stim circuit strings
derived via ``tick_circuit_to_stim()``. Supports:

- Memory experiments (syndrome extraction rounds)
- Transversal Hadamard (H on all data qubits, swaps X<->Z stabilizers)
- Transversal CNOT (CX between corresponding data qubits of two patches)
- Transversal SZ via gate teleportation (CX + |+Y> ancilla consumption)

Output formats:

- ``to_tick_circuit()`` -- PECOS TickCircuit (source of truth)
- ``to_dag_circuit()`` -- PECOS DagCircuit (for fault analysis)
- ``to_stim()`` -- Stim circuit string (derived from TickCircuit)
- ``build_dem()`` -- DEM via PECOS DagFaultAnalyzer (no Stim)
- ``build_decoder()`` -- integrated decoder pipeline

References:
- Geher et al., "Error-corrected Hadamard gate" (arXiv:2312.11605)
- Sahay et al., "Error correction of transversal CNOT" (arXiv:2408.01393)
- Serra-Peralta et al., "Decoding across transversal Clifford gates" (arXiv:2505.13599)
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import cache, lru_cache
from itertools import zip_longest
from typing import TYPE_CHECKING

from pecos_rslib.qec import DEM_SLICE_ROUND_ATTRIBUTE, transform_two_patch_pauli

from pecos.qec.surface import gadgets
from pecos.qec.surface.circuit_builder import OpType, QubitAllocation

if TYPE_CHECKING:
    from pecos.qec.surface.circuit_builder import SurfaceCircuitStep
    from pecos.qec.surface.patch import Stabilizer, SurfacePatch

PatchSnapshot = dict[str, bool]

_SURFACE_DEM_SLICE_CACHE_SIZE = 16


@dataclass(frozen=True)
class _CachedSurfaceMemoryDemSlices:
    """Noise-weighted bounded cached slices for one physical memory family."""

    output_model: object
    initialization: object
    bulk: object
    pre_terminal: object
    terminal: object


@dataclass(frozen=True)
class _CachedSurfaceSingletonMemoryDemSlices:
    """Noise-weighted cached slices for a one-SEC-round memory experiment."""

    output_model: object
    initialization: object
    terminal: object


@dataclass(frozen=True)
class _CachedSurfaceMultiMemoryDemSlices:
    """Canonical bounded cached slices for simultaneous independent patches."""

    output_model: object
    initialization: object
    bulk: object | None
    pre_terminal: object | None
    terminal: object
    stream_counts: tuple[int, ...]
    coordinate_origins: tuple[tuple[float, float], ...]


@lru_cache(maxsize=_SURFACE_DEM_SLICE_CACHE_SIZE)
def _cached_surface_memory_dem_slices(
    dx: int,
    dz: int,
    orientation_name: str,
    rotated: bool,
    basis: str,
    p1: float,
    p2: float,
    p_meas: float,
    p_prep: float,
) -> _CachedSurfaceMemoryDemSlices:
    """Compile the constant-depth surface-memory physical fixture on a cache miss."""
    from pecos.qec.surface.patch import PatchOrientation, SurfacePatch

    patch = SurfacePatch.create(
        dx=dx,
        dz=dz,
        orientation=PatchOrientation[orientation_name],
        rotated=rotated,
    )
    builder = LogicalCircuitBuilder()
    builder.add_patch(patch, "fixture", coord_offset=(0.0, 0.0))
    builder.add_memory("fixture", rounds=3, basis=basis)
    model, influence_map, dag_circuit = builder._build_structured_dem(  # noqa: SLF001
        p1=p1,
        p2=p2,
        p_meas=p_meas,
        p_prep=p_prep,
    )
    schedule = model.round_schedule(influence_map, dag_circuit)
    return _CachedSurfaceMemoryDemSlices(
        output_model=model,
        initialization=schedule.cached_slice(0),
        bulk=schedule.cached_slice(1),
        pre_terminal=schedule.cached_slice(2),
        terminal=schedule.cached_slice(3),
    )


@lru_cache(maxsize=_SURFACE_DEM_SLICE_CACHE_SIZE)
def _cached_surface_singleton_memory_dem_slices(
    dx: int,
    dz: int,
    orientation_name: str,
    rotated: bool,
    basis: str,
    p1: float,
    p2: float,
    p_meas: float,
    p_prep: float,
) -> _CachedSurfaceSingletonMemoryDemSlices:
    """Compile the constant-depth one-round surface-memory family."""
    from pecos.qec.surface.patch import PatchOrientation, SurfacePatch

    patch = SurfacePatch.create(
        dx=dx,
        dz=dz,
        orientation=PatchOrientation[orientation_name],
        rotated=rotated,
    )
    builder = LogicalCircuitBuilder()
    builder.add_patch(patch, "fixture", coord_offset=(0.0, 0.0))
    builder.add_memory("fixture", rounds=1, basis=basis)
    model, influence_map, dag_circuit = builder._build_structured_dem(  # noqa: SLF001
        p1=p1,
        p2=p2,
        p_meas=p_meas,
        p_prep=p_prep,
    )
    schedule = model.round_schedule(influence_map, dag_circuit)
    return _CachedSurfaceSingletonMemoryDemSlices(
        output_model=model,
        initialization=schedule.cached_slice(0),
        terminal=schedule.cached_slice(1),
    )


@lru_cache(maxsize=_SURFACE_DEM_SLICE_CACHE_SIZE)
def _cached_surface_multi_memory_dem_slices(
    patch_specs: tuple[tuple[int, int, str, bool], ...],
    bases: tuple[str, ...],
    singleton: bool,
    p1: float,
    p2: float,
    p_meas: float,
    p_prep: float,
) -> _CachedSurfaceMultiMemoryDemSlices:
    """Compile one canonical simultaneous-memory family.

    Patch labels, qubit offsets, and placements remain instance data. The
    canonical fixture spaces patches apart only so each stream partition has an
    unambiguous coordinate origin.
    """
    from pecos.qec.surface.patch import PatchOrientation, SurfacePatch

    builder = LogicalCircuitBuilder()
    labels = []
    coordinate_origins = []
    stream_counts = []
    qubit_offset = 0
    coordinate_x = 0.0
    for patch_index, ((dx, dz, orientation_name, rotated), basis) in enumerate(
        zip(patch_specs, bases, strict=True),
    ):
        patch = SurfacePatch.create(
            dx=dx,
            dz=dz,
            orientation=PatchOrientation[orientation_name],
            rotated=rotated,
        )
        label = f"fixture_{patch_index}"
        labels.append(label)
        coordinate_origins.append((coordinate_x, 0.0))
        builder.add_patch(
            patch,
            label,
            qubit_offset=qubit_offset,
            coord_offset=(coordinate_x, 0.0),
        )
        qubit_offset += patch.geometry.num_qubits
        coordinate_x += float(2 * dz + 4)
        if singleton:
            stabs = patch.geometry.z_stabilizers if basis == "Z" else patch.geometry.x_stabilizers
            stream_counts.append(len(stabs))
        else:
            stream_counts.append(len(patch.geometry.x_stabilizers) + len(patch.geometry.z_stabilizers))

    builder.add_memory(
        labels,
        rounds=1 if singleton else 3,
        basis=dict(zip(labels, bases, strict=True)),
    )
    model, influence_map, dag_circuit = builder._build_structured_dem(  # noqa: SLF001
        p1=p1,
        p2=p2,
        p_meas=p_meas,
        p_prep=p_prep,
    )
    schedule = model.round_schedule(influence_map, dag_circuit)
    return _CachedSurfaceMultiMemoryDemSlices(
        output_model=model,
        initialization=schedule.cached_slice(0),
        bulk=None if singleton else schedule.cached_slice(1),
        pre_terminal=None if singleton else schedule.cached_slice(2),
        terminal=schedule.cached_slice(1 if singleton else 3),
        stream_counts=tuple(stream_counts),
        coordinate_origins=tuple(coordinate_origins),
    )


@dataclass(frozen=True)
class _CachedSurfaceBoundaryDemSlices:
    """Noise-weighted cached slices around one bounded logical-gate boundary."""

    output_model: object
    initialization: object
    pre_gate_bulk: object
    pre_gate_boundary: object
    gate_boundary: object
    post_gate_bulk: object
    pre_terminal: object
    terminal: object


@lru_cache(maxsize=_SURFACE_DEM_SLICE_CACHE_SIZE)
def _cached_surface_h_dem_slices(
    dx: int,
    dz: int,
    orientation_name: str,
    rotated: bool,
    initial_basis: str,
    final_basis: str,
    p1: float,
    p2: float,
    p_meas: float,
    p_prep: float,
    *,
    pre_gate_swapped: bool = False,
    future_h_parity: bool = False,
) -> _CachedSurfaceBoundaryDemSlices:
    """Compile one physical/logical-frame H-boundary family on a cache miss.

    An optional earlier H establishes the physical X/Z assignment entering the
    selected boundary. An optional later H represents the parity of all H gates
    after it when the final observable is propagated backwards. Three SEC rounds
    on either side isolate the selected boundary from those auxiliary gates.
    """
    from pecos.qec.surface.patch import PatchOrientation, SurfacePatch

    patch = SurfacePatch.create(
        dx=dx,
        dz=dz,
        orientation=PatchOrientation[orientation_name],
        rotated=rotated,
    )
    builder = LogicalCircuitBuilder()
    builder.add_patch(patch, "fixture", coord_offset=(0.0, 0.0))
    builder.add_memory("fixture", rounds=3, basis=initial_basis)
    if pre_gate_swapped:
        builder.add_transversal_h("fixture")
        builder.add_memory("fixture", rounds=3, basis=initial_basis)
    selected_boundary_round = 3 + 3 * int(pre_gate_swapped)
    builder.add_transversal_h("fixture")
    builder.add_memory("fixture", rounds=3, basis=final_basis)
    if future_h_parity:
        builder.add_transversal_h("fixture")
        builder.add_memory("fixture", rounds=3, basis=final_basis)
    model, influence_map, dag_circuit = builder._build_structured_dem(  # noqa: SLF001
        p1=p1,
        p2=p2,
        p_meas=p_meas,
        p_prep=p_prep,
    )
    schedule = model.round_schedule(influence_map, dag_circuit)
    terminal_round = 6 + 3 * int(pre_gate_swapped) + 3 * int(future_h_parity)
    return _CachedSurfaceBoundaryDemSlices(
        output_model=model,
        initialization=schedule.cached_slice(0),
        pre_gate_bulk=schedule.cached_slice(selected_boundary_round - 2),
        pre_gate_boundary=schedule.cached_slice(selected_boundary_round - 1),
        gate_boundary=schedule.cached_slice(selected_boundary_round),
        post_gate_bulk=schedule.cached_slice(selected_boundary_round + 1),
        pre_terminal=schedule.cached_slice(terminal_round - 1),
        terminal=schedule.cached_slice(terminal_round),
    )


@dataclass(frozen=True)
class _CachedSurfaceCxDemSlices:
    """Bounded transversal-CX cached slices and their canonical stream layout."""

    cached_slices: _CachedSurfaceBoundaryDemSlices
    control_stream_count: int
    target_stream_count: int
    target_coordinate_origin: tuple[float, float]


def _boundary_slice_placements(
    boundary_slices: list[_CachedSurfaceBoundaryDemSlices],
    memory_rounds: list[int],
) -> tuple[list[tuple[object, int, int]], int]:
    """Place boundary families and retain the family index for output routing."""
    if not boundary_slices or len(memory_rounds) != len(boundary_slices) + 1:
        msg = "boundary cached slices require exactly one more memory segment"
        raise ValueError(msg)

    first_boundary_round = memory_rounds[0]
    placements = [(boundary_slices[0].initialization, 0, 0)]
    placements.extend((boundary_slices[0].pre_gate_bulk, round_, 0) for round_ in range(1, first_boundary_round - 1))
    boundary_round = first_boundary_round
    for boundary_index, cached_slices in enumerate(boundary_slices):
        placements.extend(
            [
                (cached_slices.pre_gate_boundary, boundary_round - 1, boundary_index),
                (cached_slices.gate_boundary, boundary_round, boundary_index),
            ],
        )
        next_boundary_round = boundary_round + memory_rounds[boundary_index + 1]
        if boundary_index + 1 < len(boundary_slices):
            next_slices = boundary_slices[boundary_index + 1]
            placements.extend(
                (next_slices.pre_gate_bulk, round_, boundary_index + 1)
                for round_ in range(boundary_round + 1, next_boundary_round - 1)
            )
        else:
            placements.extend(
                (cached_slices.post_gate_bulk, round_, boundary_index)
                for round_ in range(boundary_round + 1, next_boundary_round - 1)
            )
            placements.extend(
                [
                    (cached_slices.pre_terminal, next_boundary_round - 1, boundary_index),
                    (cached_slices.terminal, next_boundary_round, boundary_index),
                ],
            )
        boundary_round = next_boundary_round
    return placements, boundary_round


def _two_patch_detector_coordinate_offsets(
    control_offset: tuple[float, float],
    target_offset: tuple[float, float],
    *,
    control_stream_count: int,
    target_stream_count: int,
    target_coordinate_origin: tuple[float, float],
) -> dict[int, tuple[float, float]]:
    """Translate one canonical two-patch stream layout to its live placements."""
    control_x, control_y = control_offset
    target_x, target_y = target_offset
    target_origin_x, target_origin_y = target_coordinate_origin
    offsets = {stream: (float(control_x), float(control_y)) for stream in range(control_stream_count)}
    offsets.update(
        {
            stream: (float(target_x) - target_origin_x, float(target_y) - target_origin_y)
            for stream in range(
                control_stream_count,
                control_stream_count + target_stream_count,
            )
        },
    )
    return offsets


class _BoundaryOutputRouting(Enum):
    """Finite GF(2) routing policies understood by the common assembler."""

    IDENTITY = auto()
    REPEATED_CX = auto()


@dataclass(frozen=True)
class _BoundaryDemProviderDescription:
    """Complete assembly description produced by a boundary-family provider.

    Gate-specific code is responsible for eligibility and bounded physical
    compilation. Once this value exists, cached slice placement, detector
    relocation, output routing, schema validation, and composition are shared.
    Eligibility remains the provider's responsibility; this value contains
    only data consumed by the common assembler.
    """

    boundary_slices: tuple[_CachedSurfaceBoundaryDemSlices, ...]
    memory_rounds: tuple[int, ...]
    output_routing: _BoundaryOutputRouting
    coordinate_offset: tuple[float, float] | None = None
    detector_coordinate_offsets: dict[int, tuple[float, float]] | None = None
    final_basis: str | None = None


def _identity_boundary_output_routings(
    placements: list[tuple[object, int, int]],
) -> dict[int, dict[int, list[int]]]:
    return {
        round_: {output: [output] for output in cached_slice.dem_outputs}
        for cached_slice, round_, _ in placements
        if cached_slice.dem_outputs
    }


def _boundary_output_routings(
    provider: _BoundaryDemProviderDescription,
    placements: list[tuple[object, int, int]],
) -> dict[int, dict[int, list[int]]]:
    if provider.output_routing is _BoundaryOutputRouting.IDENTITY:
        return _identity_boundary_output_routings(placements)

    gate_count = len(provider.boundary_slices)
    routings = {}
    for cached_slice, round_, boundary_index in placements:
        if not cached_slice.dem_outputs:
            continue
        later_gate_count = gate_count - boundary_index - 1
        if later_gate_count % 2 == 0:
            routing = {output: [output] for output in cached_slice.dem_outputs}
        else:
            if provider.final_basis not in {"X", "Z"}:
                msg = "repeated-CX output routing requires a final X or Z basis"
                raise ValueError(msg)
            if provider.final_basis == "X":
                routing = {output: [0] if output == 0 else [0, 1] for output in cached_slice.dem_outputs}
            else:
                routing = {output: [0, 1] if output == 0 else [1] for output in cached_slice.dem_outputs}
        routings[round_] = routing
    return routings


_TWO_PATCH_IDENTITY = (1, 2, 4, 8)
_TWO_PATCH_GATES = ("h0", "h1", "cx")


def _append_two_patch_gate_transform(transform: tuple[int, ...], gate: str) -> tuple[int, ...]:
    """Append ``gate`` to a chronological Pauli-frame transformation."""
    return tuple(transform_two_patch_pauli(pauli, gate) for pauli in transform)


@cache
def _canonical_two_patch_suffix(
    start_swapped: tuple[bool, bool],
    target_transform: tuple[int, ...],
    target_swapped: tuple[bool, bool],
) -> tuple[str, ...]:
    """Find a bounded canonical word for a future H/CX logical action.

    The search state includes physical X/Z orientation because the surface
    frontend permits CX only when both patches have the same orientation.
    There are finitely many two-qubit real-Clifford transformations, so this
    normalization keeps cached slice identity independent of algorithm depth.
    """
    start = (_TWO_PATCH_IDENTITY, start_swapped)
    target = (target_transform, target_swapped)
    queue = deque([(start, ())])
    seen = {start}
    while queue:
        (transform, swapped), word = queue.popleft()
        if (transform, swapped) == target:
            return word
        for gate in _TWO_PATCH_GATES:
            if gate == "cx" and swapped[0] != swapped[1]:
                continue
            next_swapped = swapped
            if gate == "h0":
                next_swapped = (not swapped[0], swapped[1])
            elif gate == "h1":
                next_swapped = (swapped[0], not swapped[1])
            next_state = (_append_two_patch_gate_transform(transform, gate), next_swapped)
            if next_state not in seen:
                seen.add(next_state)
                queue.append((next_state, (*word, gate)))
    msg = "future H/CX action is unreachable under the surface orientation constraint"
    raise ValueError(msg)


@lru_cache(maxsize=_SURFACE_DEM_SLICE_CACHE_SIZE)
def _cached_surface_cx_dem_slices(
    control_dx: int,
    control_dz: int,
    control_orientation_name: str,
    control_rotated: bool,
    target_dx: int,
    target_dz: int,
    target_orientation_name: str,
    target_rotated: bool,
    initial_control_basis: str,
    initial_target_basis: str,
    final_control_basis: str,
    final_target_basis: str,
    p1: float,
    p2: float,
    p_meas: float,
    p_prep: float,
) -> _CachedSurfaceCxDemSlices:
    """Compile a bounded two-patch memory-CX-memory physical fixture."""
    from pecos.qec.surface.patch import PatchOrientation, SurfacePatch

    control = SurfacePatch.create(
        dx=control_dx,
        dz=control_dz,
        orientation=PatchOrientation[control_orientation_name],
        rotated=control_rotated,
    )
    target = SurfacePatch.create(
        dx=target_dx,
        dz=target_dz,
        orientation=PatchOrientation[target_orientation_name],
        rotated=target_rotated,
    )
    target_origin = (float(max(control_dz, target_dz) * 2 + 2), 0.0)
    builder = LogicalCircuitBuilder()
    builder.add_patch(control, "control", coord_offset=(0.0, 0.0))
    builder.add_patch(
        target,
        "target",
        qubit_offset=control.geometry.num_qubits,
        coord_offset=target_origin,
    )
    builder.add_memory(
        ["control", "target"],
        rounds=3,
        basis={"control": initial_control_basis, "target": initial_target_basis},
    )
    builder.add_transversal_cx("control", "target")
    builder.add_memory(
        ["control", "target"],
        rounds=3,
        basis={"control": final_control_basis, "target": final_target_basis},
    )
    model, influence_map, dag_circuit = builder._build_structured_dem(  # noqa: SLF001
        p1=p1,
        p2=p2,
        p_meas=p_meas,
        p_prep=p_prep,
    )
    schedule = model.round_schedule(influence_map, dag_circuit)
    cached_slices = _CachedSurfaceBoundaryDemSlices(
        output_model=model,
        initialization=schedule.cached_slice(0),
        pre_gate_bulk=schedule.cached_slice(1),
        pre_gate_boundary=schedule.cached_slice(2),
        gate_boundary=schedule.cached_slice(3),
        post_gate_bulk=schedule.cached_slice(4),
        pre_terminal=schedule.cached_slice(5),
        terminal=schedule.cached_slice(6),
    )
    control_stream_count = len(control.geometry.x_stabilizers) + len(control.geometry.z_stabilizers)
    target_stream_count = len(target.geometry.x_stabilizers) + len(target.geometry.z_stabilizers)
    return _CachedSurfaceCxDemSlices(
        cached_slices=cached_slices,
        control_stream_count=control_stream_count,
        target_stream_count=target_stream_count,
        target_coordinate_origin=target_origin,
    )


@lru_cache(maxsize=_SURFACE_DEM_SLICE_CACHE_SIZE)
def _cached_surface_mixed_dem_slices(
    control_dx: int,
    control_dz: int,
    control_orientation_name: str,
    control_rotated: bool,
    target_dx: int,
    target_dz: int,
    target_orientation_name: str,
    target_rotated: bool,
    initial_control_basis: str,
    initial_target_basis: str,
    final_control_basis: str,
    final_target_basis: str,
    selected_gate: str,
    pre_gate_swapped: tuple[bool, bool],
    future_word: tuple[str, ...],
    p1: float,
    p2: float,
    p_meas: float,
    p_prep: float,
) -> _CachedSurfaceCxDemSlices:
    """Compile one normalized boundary from a mixed two-patch H/CX schedule."""
    from pecos.qec.surface.patch import PatchOrientation, SurfacePatch

    control = SurfacePatch.create(
        dx=control_dx,
        dz=control_dz,
        orientation=PatchOrientation[control_orientation_name],
        rotated=control_rotated,
    )
    target = SurfacePatch.create(
        dx=target_dx,
        dz=target_dz,
        orientation=PatchOrientation[target_orientation_name],
        rotated=target_rotated,
    )
    target_origin = (float(max(control_dz, target_dz) * 2 + 2), 0.0)
    builder = LogicalCircuitBuilder()
    builder.add_patch(control, "control", coord_offset=(0.0, 0.0))
    builder.add_patch(
        target,
        "target",
        qubit_offset=control.geometry.num_qubits,
        coord_offset=target_origin,
    )
    both = ["control", "target"]
    final_bases = {"control": final_control_basis, "target": final_target_basis}
    builder.add_memory(
        both,
        rounds=3,
        basis={"control": initial_control_basis, "target": initial_target_basis},
    )
    if any(pre_gate_swapped):
        if pre_gate_swapped[0]:
            builder.add_transversal_h("control")
        if pre_gate_swapped[1]:
            builder.add_transversal_h("target")
        builder.add_memory(both, rounds=3, basis=final_bases)
    selected_boundary_round = 3 + 3 * int(any(pre_gate_swapped))

    def append_gate(gate: str) -> None:
        if gate == "h0":
            builder.add_transversal_h("control")
        elif gate == "h1":
            builder.add_transversal_h("target")
        elif gate == "cx":
            builder.add_transversal_cx("control", "target")
        else:  # pragma: no cover - cache callers validate the alphabet
            msg = f"unknown normalized gate {gate!r}"
            raise ValueError(msg)

    append_gate(selected_gate)
    builder.add_memory(both, rounds=3, basis=final_bases)
    for gate in future_word:
        append_gate(gate)
        builder.add_memory(both, rounds=3, basis=final_bases)

    model, influence_map, dag_circuit = builder._build_structured_dem(  # noqa: SLF001
        p1=p1,
        p2=p2,
        p_meas=p_meas,
        p_prep=p_prep,
    )
    schedule = model.round_schedule(influence_map, dag_circuit)
    terminal_round = selected_boundary_round + 3 * (1 + len(future_word))
    cached_slices = _CachedSurfaceBoundaryDemSlices(
        output_model=model,
        initialization=schedule.cached_slice(0),
        pre_gate_bulk=schedule.cached_slice(selected_boundary_round - 2),
        pre_gate_boundary=schedule.cached_slice(selected_boundary_round - 1),
        gate_boundary=schedule.cached_slice(selected_boundary_round),
        post_gate_bulk=schedule.cached_slice(selected_boundary_round + 1),
        pre_terminal=schedule.cached_slice(terminal_round - 1),
        terminal=schedule.cached_slice(terminal_round),
    )
    control_stream_count = len(control.geometry.x_stabilizers) + len(control.geometry.z_stabilizers)
    target_stream_count = len(target.geometry.x_stabilizers) + len(target.geometry.z_stabilizers)
    return _CachedSurfaceCxDemSlices(
        cached_slices=cached_slices,
        control_stream_count=control_stream_count,
        target_stream_count=target_stream_count,
        target_coordinate_origin=target_origin,
    )


def _validate_boundary_cardinality(segments: list[object], boundary_gates: list[object]) -> None:
    """Require exactly one boundary list between consecutive segments."""
    expected_boundaries = len(segments) - 1
    if len(boundary_gates) != expected_boundaries:
        msg = (
            f"algorithm descriptor has {len(boundary_gates)} boundary gate lists and "
            f"{len(segments)} segments; expected exactly one boundary list between "
            "consecutive segments"
        )
        raise ValueError(msg)


class LogicalGateType(Enum):
    """Types of logical operations in a surface code circuit."""

    MEMORY = auto()
    TRANSVERSAL_H = auto()
    TRANSVERSAL_SZ = auto()
    TRANSVERSAL_SZdg = auto()
    TRANSVERSAL_CX = auto()


@dataclass
class PatchState:
    """Tracks the stabilizer assignment state of a patch.

    After transversal H, X-stabilizers become Z-stabilizers and vice versa.
    This state tracks which physical stabilizers are currently X-type vs Z-type,
    so that detectors can be formed correctly across gate boundaries.
    """

    patch: SurfacePatch
    label: str
    qubit_offset: int = 0
    coord_offset: tuple[float, float] = (0.0, 0.0)
    x_z_swapped: bool = False
    stabilizers_by_index: dict[str, dict[int, Stabilizer]] = field(default_factory=dict)

    @property
    def current_x_stabilizers(self) -> list[Stabilizer]:
        """Stabilizers currently measuring X-type checks."""
        if self.x_z_swapped:
            return self.patch.geometry.z_stabilizers
        return self.patch.geometry.x_stabilizers

    @property
    def current_z_stabilizers(self) -> list[Stabilizer]:
        """Stabilizers currently measuring Z-type checks."""
        if self.x_z_swapped:
            return self.patch.geometry.x_stabilizers
        return self.patch.geometry.z_stabilizers


@dataclass
class LogicalOp:
    """A logical operation in the circuit."""

    gate_type: LogicalGateType
    patches: list[str]
    rounds: int = 0
    basis: str = "Z"
    per_patch_basis: dict[str, str] = field(default_factory=dict)
    # Teleportation consumes the target as a correction readout, separate
    # from deterministic observables on the data.
    teleportation: bool = False
    # Type of magic state injection: "T" for T-gate, "SZ" for SZ, or None.
    # Used by build_algorithm_descriptor() to emit the correct boundary gate.
    injection_type: str | None = None


def _conjugate_pauli(op: LogicalOp, patch: str, pauli: str) -> list[tuple[str, str]]:
    """Return the sign-free Pauli image under a transversal Clifford layer."""
    if patch not in op.patches:
        return [(patch, pauli)]
    if op.gate_type == LogicalGateType.TRANSVERSAL_H:
        return [(patch, {"X": "Z", "Y": "Y", "Z": "X"}[pauli])]
    if op.gate_type in (LogicalGateType.TRANSVERSAL_SZ, LogicalGateType.TRANSVERSAL_SZdg):
        return [(patch, {"X": "Y", "Y": "X", "Z": "Z"}[pauli])]
    terms = [(patch, pauli)]
    if op.gate_type == LogicalGateType.TRANSVERSAL_CX:
        ctrl, tgt = op.patches
        if patch == ctrl and pauli in {"X", "Y"}:
            terms.append((tgt, "X"))
        elif patch == tgt and pauli in {"Z", "Y"}:
            terms.append((ctrl, "Z"))
    return terms


def _conjugate_stabilizer_term(
    op: LogicalOp,
    term: tuple[str, str, str],
) -> list[tuple[str, str, str]]:
    """Conjugate an even-weight check while retaining its physical register support."""
    patch, base_family, pauli = term
    return [(label, base_family, kind) for label, kind in _conjugate_pauli(op, patch, pauli)]


@dataclass(frozen=True)
class _PropagationContext:
    """Forward memory and orientation information shared by all check walks."""

    operations: tuple[LogicalOp, ...]
    memory_indices: tuple[int, ...]
    orientations: tuple[dict[str, bool], ...]
    preparations: dict[str, tuple[int, str]]

    @classmethod
    def from_operations(cls, operations: list[LogicalOp]) -> _PropagationContext:
        """Index preparations and physical register orientations once per build."""
        memory_indices = []
        orientations = []
        swapped: dict[str, bool] = {}
        preparations: dict[str, tuple[int, str]] = {}
        for operation_index, op in enumerate(operations):
            if op.gate_type == LogicalGateType.TRANSVERSAL_H:
                for label in op.patches:
                    swapped[label] = not swapped.get(label, False)
            elif op.gate_type == LogicalGateType.MEMORY:
                segment = len(memory_indices)
                memory_indices.append(operation_index)
                orientations.append({label: swapped.get(label, False) for label in op.patches})
                for label in op.patches:
                    preparations.setdefault(label, (segment, op.per_patch_basis.get(label, op.basis)))
        return cls(tuple(operations), tuple(memory_indices), tuple(orientations), preparations)


def _propagate_stabilizer_terms(
    context: _PropagationContext,
    segment_idx: int,
    term: tuple[str, str, str],
) -> list[tuple[str, str, int]] | None:
    """Resolve a (patch, base family, Pauli) to ordered (patch, family, segment) records.

    Terms form an insertion-ordered XOR set on fixed physical supports. A
    positive-round memory resolves only the type that its register measured.
    A preparation closes a matching Pauli with its known +1 sign, including
    the current segment's own preparation. None means no detector: the Pauli
    was not measured, or the preparation it reached is of another type, so the
    model treats it as random. Reaching the start without preparation raises
    ValueError naming the unresolved terms.

    Known limitations, each losing detectors rather than emitting a wrong one
    (every emitted detector is deterministic and per-observable fault distance
    is unchanged in the probed shapes):

    - d=2 ``M(2,Z); H; SZ; M(2,X)``: Y is determined by measured X/Z products
      beyond this single-support model.
    - A term that dead-ends at a partner's preparation of a different type is
      treated as random even when the partner's same-round measurement would
      close it: ``M(A,2,Z); M(B,0,Z); CX(A,B); M([A,B],2,Z)`` misses four
      weight-3 detectors X_A(seg1,r0) X_B(seg1,r0) X_A(seg0,last).
    - A term is never resolved at a measurement later than the gate that
      created it: ``M(A,0,X); M(B,2,Z); CX(A,B); M(A,2,Z); M([A,B],2,Z)``
      misses Z_B(seg3,r0) Z_A(seg2,r0) Z_B(seg1,last).
    - Products of two current checks through CX;S;CX:
      ``M([A,B],2,Z); CX(A,B); SZ(B); CX(A,B); M([A,B],2,Z)`` misses the
      weight-4 X_A X_B products.
    """
    preparations = context.preparations
    patch, _, pauli = term
    if patch in preparations and preparations[patch][0] == segment_idx:
        return [] if pauli == preparations[patch][1] else None
    open_terms = dict.fromkeys([term])
    resolved = []
    previous_segment = segment_idx
    for operation_index in range(context.memory_indices[segment_idx] - 1, -1, -1):
        op = context.operations[operation_index]
        if op.gate_type == LogicalGateType.MEMORY:
            previous_segment -= 1
            for term in tuple(open_terms):
                patch, base_family, pauli = term
                if patch in op.patches:
                    if op.rounds > 0:
                        measured_type = base_family
                        if context.orientations[previous_segment][patch]:
                            measured_type = "Z" if base_family == "X" else "X"
                        if pauli != measured_type:
                            return None
                        resolved.append((patch, pauli, previous_segment))
                        del open_terms[term]
                    elif preparations[patch][0] == previous_segment:
                        if pauli != preparations[patch][1]:
                            return None
                        del open_terms[term]
            if not open_terms:
                break
            continue

        transformed: dict[tuple[str, str, str], None] = {}
        for term in open_terms:
            for mapped_term in _conjugate_stabilizer_term(op, term):
                if mapped_term in transformed:
                    del transformed[mapped_term]
                else:
                    transformed[mapped_term] = None
        open_terms = transformed
    if open_terms:
        msg = f"Stabilizer terms without preparation: {list(open_terms)}"
        raise ValueError(msg)
    return resolved


def _logical_readout_is_deterministic(
    operations: list[LogicalOp],
    segment_idx: int,
    patch: str,
    logical_type: str,
) -> bool:
    """Propagate a final logical readout backwards to product preparations.

    ``logical_type`` is X or Z in the readout's current orientation (its
    measurement basis). Each crossed H swaps the type during the walk; callers
    must not also swap the initial type. Physical S/S-dagger preserves Z but
    has no supported logical X image, including the unmodelled distance-1 case.
    A dict provides insertion-ordered XOR terms, so repeated CX images cancel.
    Terms involving a patch consumed before a crossed gate are unreliable.
    This deliberately conservative rule suppresses B and C for
    M([A,B,C],2,Z); CX(A,B); CX(B,C); M([B,C],2,Z), even though the simulator's
    non-destructive MZ makes them deterministic, because A is gated after its
    final readout, an invalid program rejected by ``to_tick_circuit``.
    Terms without preparation raise ValueError naming the terms. False means
    the readout is not supported as deterministic; unlike the check walk,
    which returns None exactly for an unmeasured (physically random) Pauli,
    this also includes the unsupported logical X image under physical S.
    """
    if logical_type not in {"X", "Z"}:
        msg = f"Unsupported logical readout type {logical_type!r}; expected X or Z"
        raise ValueError(msg)
    memory_indices = []
    first_memory = {}
    last_memory = {}
    for index, op in enumerate(operations):
        if op.gate_type == LogicalGateType.MEMORY:
            memory_indices.append(index)
            for label in op.patches:
                first_memory.setdefault(label, index)
                last_memory[label] = index
    if not 0 <= segment_idx < len(memory_indices):
        msg = f"No memory segment {segment_idx} for patch '{patch}'"
        raise ValueError(msg)
    start = memory_indices[segment_idx]
    if patch not in operations[start].patches:
        msg = f"Patch '{patch}' is not in memory segment {segment_idx}"
        raise ValueError(msg)

    terms = {(patch, logical_type): None}
    for index in range(start, -1, -1):
        op = operations[index]
        if op.gate_type == LogicalGateType.MEMORY:
            for label, kind in tuple(terms):
                if first_memory.get(label) == index:
                    if op.per_patch_basis.get(label, op.basis) != kind:
                        return False
                    del terms[label, kind]
            if not terms:
                return True
            continue

        images = {}
        for label, kind in terms:
            if (
                op.gate_type in {LogicalGateType.TRANSVERSAL_SZ, LogicalGateType.TRANSVERSAL_SZdg}
                and label in op.patches
                and kind == "X"
            ):
                # Physical S has no supported logical X image, although even-weight
                # checks can retain its Y image without a check-level sign.
                return False
            for term in _conjugate_pauli(op, label, kind):
                term_patch = term[0]
                if term_patch in op.patches and term_patch in last_memory and last_memory[term_patch] < index:
                    return False
                if term in images:
                    del images[term]
                else:
                    images[term] = None
        terms = images
        if not terms:
            return True

    msg = f"Logical readout for patch '{patch}' has terms without preparation: {list(terms)}"
    raise ValueError(msg)


class LogicalCircuitBuilder:
    """Builds surface code circuits with transversal gates.

    Composes logical operations on one or more patches, generating Stim
    circuits with correct detector annotations across gate boundaries.

    Example::

        patch = SurfacePatch.create(distance=3)
        builder = LogicalCircuitBuilder()
        builder.add_patch(patch, "A")
        builder.add_memory("A", rounds=3, basis="Z")
        builder.add_transversal_h("A")
        builder.add_memory("A", rounds=3, basis="X")
        stim_str = builder.to_stim(p1=0.001, p2=0.001)
    """

    def __init__(self) -> None:
        """Initialize an empty logical circuit builder."""
        self._patches: dict[str, PatchState] = {}
        self._operations: list[LogicalOp] = []
        self._consumed_injection_ancillas: set[str] = set()

    def add_patch(
        self,
        patch: SurfacePatch,
        label: str,
        qubit_offset: int = 0,
        coord_offset: tuple[float, float] | None = None,
    ) -> None:
        """Register a surface code patch.

        The geometry is validated (every check must have even weight) and
        indexed here, once; a registered patch's geometry is frozen from this
        point and later edits to it are not reflected in generated circuits.

        Args:
            patch: The surface code patch.
            label: Unique label for this patch.
            qubit_offset: Offset added to all qubit indices for this patch.
                Use this when multiple patches share a qubit index space.
            coord_offset: (dx, dy) spatial offset for this patch's
                QUBIT_COORDS and DETECTOR coordinates. If None, computed
                automatically based on patch index (patches are spaced
                apart so coordinates don't overlap).
        """
        if label in self._patches:
            msg = f"Patch '{label}' already registered"
            raise ValueError(msg)
        # S maps Y to -X, S-dagger maps X to -Y, H maps Y to -Y, each a -1
        # per qubit, so over even weight the check-level sign is (-1)^w = +1,
        # which is what lets the term model drop signs; odd weight would need signed terms.
        geometry = patch.geometry
        for family, stabs in (("X", geometry.x_stabilizers), ("Z", geometry.z_stabilizers)):
            for stab in stabs:
                if len(stab.data_qubits) % 2:
                    msg = (
                        f"Check on patch {label!r}, family {family}, index {stab.index} has odd weight; "
                        "detector propagation requires even weight"
                    )
                    raise ValueError(msg)
        if coord_offset is None:
            # Place the new patch after the right edge of every existing patch.
            # Multiplying this patch's width by its registration index can make
            # a small patch overlap a previously registered larger patch.
            next_x = max(
                (state.coord_offset[0] + state.patch.geometry.dz * 2 + 2 for state in self._patches.values()),
                default=0.0,
            )
            coord_offset = (next_x, 0.0)
        self._patches[label] = PatchState(
            patch=patch,
            label=label,
            qubit_offset=qubit_offset,
            coord_offset=coord_offset,
            stabilizers_by_index={
                "X": {stab.index: stab for stab in geometry.x_stabilizers},
                "Z": {stab.index: stab for stab in geometry.z_stabilizers},
            },
        )

    def add_memory(
        self,
        patch_labels: str | list[str],
        rounds: int,
        basis: str | dict[str, str] = "Z",
    ) -> None:
        """Add syndrome extraction rounds for one or more patches.

        When multiple patches are given, their syndrome extraction runs
        in parallel (same time window).

        Args:
            patch_labels: Label(s) of the patch(es). String for single
                patch, list for parallel multi-patch.
            rounds: Number of syndrome extraction rounds.
            basis: Measurement basis. Either a single string ('X', 'Y', 'Z')
                applied to all patches, or a dict mapping patch labels to
                their individual basis (e.g., ``{"D": "Z", "Y": "Y"}``).
                Only used for initialization and final measurement. Y is
                supported for preparation, but the final readout of a patch
                must be X or Z.
        """
        if isinstance(patch_labels, str):
            patch_labels = [patch_labels]
        for label in patch_labels:
            self._require_available_patch(label)

        if isinstance(basis, str):
            default_basis = basis.upper()
            per_patch = {}
        else:
            default_basis = "Z"
            per_patch = {k: v.upper() for k, v in basis.items()}

        for value in (default_basis, *per_patch.values()):
            if value not in {"X", "Y", "Z"}:
                msg = f"Unsupported memory basis {value!r}; expected X, Y, or Z"
                raise ValueError(msg)

        self._operations.append(
            LogicalOp(
                gate_type=LogicalGateType.MEMORY,
                patches=list(patch_labels),
                rounds=rounds,
                basis=default_basis,
                per_patch_basis=per_patch,
            ),
        )

    def _require_available_patch(self, label: str) -> None:
        if label not in self._patches:
            msg = f"Unknown patch '{label}'"
            raise ValueError(msg)
        if label in self._consumed_injection_ancillas:
            msg = f"Injection ancilla '{label}' has been consumed"
            raise ValueError(msg)

    def _require_square(self, patch_label: str, gate_name: str) -> None:
        """Check that a patch is square (dx=dz), required for transversal gates."""
        patch = self._patches[patch_label].patch
        if patch.geometry.dx != patch.geometry.dz:
            msg = f"{gate_name} requires a square patch (dx=dz), got dx={patch.geometry.dx}, dz={patch.geometry.dz}"
            raise ValueError(msg)

    def add_transversal_h(self, patch_label: str) -> None:
        """Add a transversal Hadamard gate on a patch.

        Applies H to every data qubit. After this:
        - X-stabilizers become Z-stabilizers and vice versa
        - Logical X and logical Z are exchanged
        - Detectors at the boundary compare cross-type measurements

        The patch must be square (dx=dz) for the code to remain valid.

        Args:
            patch_label: Label of the patch.
        """
        self._require_available_patch(patch_label)
        self._require_square(patch_label, "Transversal H")
        self._operations.append(
            LogicalOp(
                gate_type=LogicalGateType.TRANSVERSAL_H,
                patches=[patch_label],
            ),
        )

    def add_transversal_sz(self, patch_label: str) -> None:
        """Apply physical SZ = diag(1, i) to every data qubit of a square patch.

        This layer is not a logical S gate on this code. The future
        fold-transversal construction follows Chen, Chen, Lu, Pan
        (arXiv:2412.01391) and requires additional operations.
        """
        self._require_available_patch(patch_label)
        self._require_square(patch_label, "Transversal SZ")
        self._operations.append(
            LogicalOp(
                gate_type=LogicalGateType.TRANSVERSAL_SZ,
                patches=[patch_label],
            ),
        )

    def add_transversal_szdg(self, patch_label: str) -> None:
        """Apply physical SZdg to every data qubit of a square patch.

        This inverse physical layer is not a logical S-dagger on this code.
        The future fold-transversal construction follows Chen, Chen, Lu, Pan
        (arXiv:2412.01391) and requires additional operations.
        """
        self._require_available_patch(patch_label)
        self._require_square(patch_label, "Transversal SZdg")
        self._operations.append(
            LogicalOp(
                gate_type=LogicalGateType.TRANSVERSAL_SZdg,
                patches=[patch_label],
            ),
        )

    def add_sz_via_teleportation(
        self,
        data_label: str,
        ancilla_label: str,
        rounds_before: int = 3,
        rounds_after: int = 3,
    ) -> None:
        """Apply logical SZ via gate teleportation with |+Y> ancilla.

        Complete protocol:
        1. Prepare ancilla in |+Y> = S|+> (non-fault-tolerant injection)
        2. Syndrome rounds to project ancilla into code space
        3. Transversal CX(data=control, ancilla=target)
        4. Syndrome rounds
        5. Ancilla measured in Z-basis (final round)

        After CX, data has S|psi> (up to Z correction from ancilla outcome).
        The ancilla must have odd dx and dz for encoded logical-Y content.
        ``injection_readouts`` is emitted for a future consumer; no decoder
        applies the correction today.

        Note: The |+Y> injection is non-fault-tolerant (distance-1).
        For fault-tolerant SZ, use magic state distillation on the
        injected state before consumption.

        Args:
            data_label: Label of the data patch (receives the SZ gate).
            ancilla_label: Label of the ancilla patch (consumed).
            rounds_before: Syndrome rounds before CX.
            rounds_after: Syndrome rounds after CX.
        """
        self._require_fresh_injection_ancilla(data_label, ancilla_label)
        ancilla = self._patches[ancilla_label].patch
        if ancilla.dx % 2 == 0 or ancilla.dz % 2 == 0:
            msg = f"Injection ancilla '{ancilla_label}' requires odd dx and dz for encoded logical-Y content"
            raise ValueError(msg)
        # Step 1: Init both patches — data continues in Z, ancilla in |+Y>.
        # Per-patch basis lets us do this in a single parallel segment.
        self.add_memory(
            [data_label, ancilla_label],
            rounds=rounds_before,
            basis={data_label: "Z", ancilla_label: "Y"},
        )
        # Step 2: CX(data=control, ancilla=target) — teleports S onto data.
        # Marked as teleportation so the ancilla readout is kept separately
        # for its conditional Z correction, which preserves data Z_L.
        self._operations.append(
            LogicalOp(
                gate_type=LogicalGateType.TRANSVERSAL_CX,
                patches=[data_label, ancilla_label],
                teleportation=True,
                injection_type="SZ",
            ),
        )
        # Step 3: Post-CX extraction. Ancilla measured in Z-basis at final round.
        # If ancilla measures logical -1, apply Z correction (Pauli frame update).
        self.add_memory([data_label, ancilla_label], rounds=rounds_after, basis="Z")
        self._consumed_injection_ancillas.add(ancilla_label)

    def add_t_via_injection(
        self,
        data_label: str,
        ancilla_label: str,
        rounds_before: int = 3,
        rounds_after: int = 3,
    ) -> None:
        """Emit a Clifford stand-in for T injection using a fresh |+> ancilla.

        H on every ancilla data qubit precedes syndrome projection, transversal
        CX, and Z readout. No T gate or conditional S correction is emitted.
        The feed-forward decision point is descriptor-only; real T injection
        needs a later gadget with its own layout. ``injection_readouts`` is
        emitted for a future consumer; no decoder applies the correction today.

        Args:
            data_label: Label of the data patch.
            ancilla_label: Label of a patch not yet used in memory.
            rounds_before: Syndrome rounds before CX.
            rounds_after: Syndrome rounds after CX.
        """
        self._require_fresh_injection_ancilla(data_label, ancilla_label)
        self.add_memory(
            [data_label, ancilla_label],
            rounds=rounds_before,
            basis={data_label: "Z", ancilla_label: "X"},
        )
        # Step 2: CX(data=control, ancilla=target) in the Clifford stand-in.
        self._operations.append(
            LogicalOp(
                gate_type=LogicalGateType.TRANSVERSAL_CX,
                patches=[data_label, ancilla_label],
                teleportation=True,
                injection_type="T",
            ),
        )
        # Step 3: Post-CX extraction. Ancilla measured in Z-basis.
        # If ancilla measures logical -1 (corrected by frame), apply S.
        # This is the feed-forward decision point.
        self.add_memory(
            [data_label, ancilla_label],
            rounds=rounds_after,
            basis="Z",
        )
        self._consumed_injection_ancillas.add(ancilla_label)

    def add_transversal_cx(self, control_label: str, target_label: str) -> None:
        """Add a transversal CNOT between two patches.

        Applies CX between corresponding data qubits. After this:
        - X-errors on control propagate to target
        - Z-errors on target propagate back to control
        - Weight-3 hyperedges appear in the DEM at the gate boundary

        Both patches must have the same geometry.

        Args:
            control_label: Label of the control patch.
            target_label: Label of the target patch.
        """
        self._require_cx_geometry(control_label, target_label)
        self._operations.append(
            LogicalOp(
                gate_type=LogicalGateType.TRANSVERSAL_CX,
                patches=[control_label, target_label],
            ),
        )

    def _require_cx_geometry(self, control_label: str, target_label: str) -> None:
        for label in (control_label, target_label):
            self._require_available_patch(label)
        if not gadgets.same_static_geometry(self._patches[control_label].patch, self._patches[target_label].patch):
            msg = "Transversal CX requires the same static geometry"
            raise ValueError(msg)

    def _require_fresh_injection_ancilla(self, data_label: str, ancilla_label: str) -> None:
        self._require_cx_geometry(data_label, ancilla_label)
        if any(op.gate_type == LogicalGateType.MEMORY and ancilla_label in op.patches for op in self._operations):
            msg = f"Injection ancilla '{ancilla_label}' must be fresh (already appears in memory)"
            raise ValueError(msg)

    def _snapshot_and_reset(self) -> PatchSnapshot:
        """Snapshot patch states and reset for generation."""
        saved = {label: ps.x_z_swapped for label, ps in self._patches.items()}
        for ps in self._patches.values():
            ps.x_z_swapped = False
        return saved

    def _restore(self, saved: PatchSnapshot) -> None:
        """Restore patch states from snapshot."""
        for label, swapped in saved.items():
            self._patches[label].x_z_swapped = swapped

    def to_tick_circuit(self) -> object:
        """Generate a PECOS TickCircuit with detector and observable annotations.

        This is the primary output — the TickCircuit is the source of truth.
        Use ``to_stim()`` for Stim format (derived from TickCircuit via
        ``tick_circuit_to_stim``), or ``.to_dag_circuit()`` for fault analysis.

        Returns:
            TickCircuit with gates, detectors, and observables as metadata.
        """
        # Validate the expanded protocol: teleportation helpers insert their
        # own preparation memories before their transversal operations.
        last_memory = {
            label: index
            for index, op in enumerate(self._operations)
            if op.gate_type == LogicalGateType.MEMORY
            for label in op.patches
        }
        prepared: set[str] = set()
        for index, op in enumerate(self._operations):
            if op.gate_type == LogicalGateType.MEMORY:
                prepared.update(op.patches)
            else:
                gate_name = {
                    LogicalGateType.TRANSVERSAL_H: "Hadamard",
                    LogicalGateType.TRANSVERSAL_CX: "Cnot",
                    LogicalGateType.TRANSVERSAL_SZ: "SGate",
                    LogicalGateType.TRANSVERSAL_SZdg: "SdgGate",
                }[op.gate_type]
                for label in op.patches:
                    if label not in prepared:
                        msg = f"{gate_name} on patch {label!r} precedes that patch's first MEMORY preparation"
                        raise ValueError(msg)
                    if last_memory[label] < index:
                        msg = (
                            f"{op.gate_type.name} ({gate_name}) on patch {label!r} "
                            "executes after final data measurement; "
                            "representing this requires terminal-segment support, tracked in issue #595"
                        )
                        raise ValueError(msg)
        saved = self._snapshot_and_reset()
        gen = _CircuitGenerator(
            patches=self._patches,
            operations=self._operations,
        )
        try:
            return gen.generate()
        finally:
            self._restore(saved)

    def _assembled_dem_output_ids(self) -> list[int]:
        """Derive the emitted observable schema from the logical operation list.

        Observable IDs advance for every terminal patch measurement, including
        unreliable observables that the physical emitter omits. Injection-ancilla
        readouts do not consume an observable ID. Reliability is delegated to the
        same logical-operation walk as ``_CircuitGenerator`` so the warm cached
        slice path stays free of physical-circuit emission without duplicating
        frontend policy.
        """
        last_memory_index: dict[str, int] = {}
        for operation_index, operation in enumerate(self._operations):
            if operation.gate_type == LogicalGateType.MEMORY:
                for label in operation.patches:
                    last_memory_index[label] = operation_index

        injection_ancillas = {operation.patches[1] for operation in self._operations if operation.teleportation}

        output_ids = []
        next_output = 0
        segment_idx = 0
        for operation_index, operation in enumerate(self._operations):
            if operation.gate_type != LogicalGateType.MEMORY:
                continue
            for label in operation.patches:
                if last_memory_index.get(label) != operation_index:
                    continue
                if label in injection_ancillas:
                    continue
                basis = operation.per_patch_basis.get(label, operation.basis)
                if _logical_readout_is_deterministic(self._operations, segment_idx, label, basis):
                    output_ids.append(next_output)
                next_output += 1
            segment_idx += 1
        return output_ids

    def _assembled_detector_order_routings(self) -> dict[int, dict[int, int]]:
        """Match dense detector IDs to the frontend's orientation-aware order.

        Slice targets retain stable spatial stream identities. The gadget
        frontend emits the current logical X family before logical Z, so H
        swaps the dense ordering of the two stable physical-family ranges.
        This routing changes declaration order only; it does not relabel the
        streams used by relative targets.
        """
        stream_layout = {}
        stream_start = 0
        for label, state in self._patches.items():
            num_x = len(state.patch.geometry.x_stabilizers)
            num_z = len(state.patch.geometry.z_stabilizers)
            stream_layout[label] = (stream_start, num_x, num_z)
            stream_start += num_x + num_z

        swapped = dict.fromkeys(self._patches, False)
        round_time = 0
        routings = {}
        for operation in self._operations:
            if operation.gate_type == LogicalGateType.TRANSVERSAL_H:
                label = operation.patches[0]
                swapped[label] = not swapped[label]
                continue
            if operation.gate_type != LogicalGateType.MEMORY:
                continue

            affected_rounds = range(round_time, round_time + operation.rounds + 1)
            if any(swapped.values()):
                routing = {}
                for label, (start, num_x, num_z) in stream_layout.items():
                    if swapped[label]:
                        routing.update({start + index: start + num_z + index for index in range(num_x)})
                        routing.update({start + num_x + index: start + index for index in range(num_z)})
                    else:
                        routing.update({start + index: start + index for index in range(num_x + num_z)})
                for round_ in affected_rounds:
                    routings[round_] = routing
            else:
                # Consecutive memory segments share a boundary round. A later
                # segment after an even H parity owns that round and restores
                # the identity order.
                for round_ in affected_rounds:
                    routings.pop(round_, None)
            round_time += operation.rounds
        return routings

    def to_dag_circuit(self) -> object:
        """Generate a PECOS DagCircuit for fault analysis.

        Converts the TickCircuit to a DagCircuit, which can be used
        with ``DagFaultAnalyzer`` for fault propagation analysis.

        Returns:
            DagCircuit instance.
        """
        return self.to_tick_circuit().to_dag_circuit()

    def to_stim(
        self,
        *,
        p1: float = 0.0,
        p2: float = 0.0,
        p_meas: float = 0.0,
        p_prep: float = 0.0,
    ) -> str:
        """Generate a Stim circuit string with correct detectors.

        Builds a TickCircuit (source of truth), then converts to Stim
        format with noise injection via ``tick_circuit_to_stim()``.

        Args:
            p1: Single-qubit depolarizing error rate.
            p2: Two-qubit depolarizing error rate.
            p_meas: Measurement error rate.
            p_prep: Preparation error rate.

        Returns:
            Stim circuit string.
        """
        from pecos.qec.surface.circuit_builder import tick_circuit_to_stim

        tc = self.to_tick_circuit()
        return tick_circuit_to_stim(tc, p1=p1, p2=p2, p_meas=p_meas, p_prep=p_prep)

    def stab_coords(self) -> list[dict[str, list[tuple[float, float]]]]:
        """Compute stabilizer coordinates for all patches.

        Returns a list (one per patch, in registration order) of dicts
        with keys "X" and "Z" mapping to ancilla (x, y) positions.
        These coordinates match the detector annotations in the Stim circuit.

        Used as input to ``LogicalSubgraphDecoder``.
        """
        result = []
        for ps in self._patches.values():
            geom = ps.patch.geometry
            cx, cy = ps.coord_offset
            x_coords = []
            for s in geom.x_stabilizers:
                positions = [geom.id_to_pos[q] for q in s.data_qubits]
                avg_row = sum(r for r, c in positions) / len(positions)
                avg_col = sum(c for r, c in positions) / len(positions)
                x_coords.append((avg_col * 2 + cx, avg_row * 2 + cy))
            z_coords = []
            for s in geom.z_stabilizers:
                positions = [geom.id_to_pos[q] for q in s.data_qubits]
                avg_row = sum(r for r, c in positions) / len(positions)
                avg_col = sum(c for r, c in positions) / len(positions)
                z_coords.append((avg_col * 2 + cx, avg_row * 2 + cy))
            result.append({"X": x_coords, "Z": z_coords})
        return result

    def _build_structured_dem(
        self,
        *,
        p1: float = 0.001,
        p2: float = 0.001,
        p_meas: float = 0.001,
        p_prep: float = 0.0,
    ) -> tuple[object, object, object]:
        """Build the native DEM together with its source-tracking context."""
        from pecos_rslib.qec import DagFaultAnalyzer, DemBuilder

        tc = self.to_tick_circuit()
        dc = tc.to_dag_circuit()
        analyzer = DagFaultAnalyzer(dc)
        influence_map = analyzer.build_influence_map()

        det_json = tc.get_meta("detectors")
        obs_json = tc.get_meta("observables")
        num_meas = int(tc.get_meta("num_measurements"))

        meas_order = []
        for tick_idx in range(tc.num_ticks()):
            tick = tc.get_tick(tick_idx)
            for gate in tick.gate_batches():
                if gate.gate_type.name == "MZ":
                    meas_order.extend(int(q) for q in gate.qubits)

        dem_builder = DemBuilder(influence_map)
        dem_builder = dem_builder.with_noise(p1, p2, p_meas, p_prep)
        dem_builder = dem_builder.with_detectors_json(det_json)
        dem_builder = dem_builder.with_observables_json(obs_json)
        dem_builder = dem_builder.with_num_measurements(num_meas)
        dem_builder = dem_builder.with_measurement_order(meas_order)

        return dem_builder.build(), influence_map, dc

    def _build_structured_dem_from_cached_slices(
        self,
        *,
        p1: float,
        p2: float,
        p_meas: float,
        p_prep: float,
    ) -> tuple[object, object] | None:
        """Assemble an eligible surface DEM from bounded slice caches."""
        cached_multi_memory = self._build_structured_multi_memory_dem_from_cached_slices(
            p1=p1,
            p2=p2,
            p_meas=p_meas,
            p_prep=p_prep,
        )
        if cached_multi_memory is not None:
            return cached_multi_memory
        if len(self._patches) == 1 and len(self._operations) >= 3:
            cached_h = self._build_structured_h_dem_from_cached_slices(
                p1=p1,
                p2=p2,
                p_meas=p_meas,
                p_prep=p_prep,
            )
            if cached_h is not None:
                return cached_h
        if len(self._patches) == 2 and len(self._operations) >= 3:
            cached_mixed = self._build_structured_mixed_dem_from_cached_slices(
                p1=p1,
                p2=p2,
                p_meas=p_meas,
                p_prep=p_prep,
            )
            if cached_mixed is not None:
                return cached_mixed
            cached_cx = self._build_structured_cx_dem_from_cached_slices(
                p1=p1,
                p2=p2,
                p_meas=p_meas,
                p_prep=p_prep,
            )
            if cached_cx is not None:
                return cached_cx
        if len(self._patches) != 1 or len(self._operations) != 1:
            return None

        operation = self._operations[0]
        if operation.gate_type != LogicalGateType.MEMORY or len(operation.patches) != 1 or operation.rounds < 1:
            return None

        patch_label = operation.patches[0]
        patch_state = self._patches[patch_label]
        geometry = patch_state.patch.geometry
        basis = operation.per_patch_basis.get(patch_label, operation.basis).upper()
        coord_x, coord_y = patch_state.coord_offset
        from pecos_rslib.qec import DemSliceRoundSchedule

        if operation.rounds == 1:
            cached_slices = _cached_surface_singleton_memory_dem_slices(
                geometry.dx,
                geometry.dz,
                geometry.orientation.name,
                geometry.rotated,
                basis,
                p1,
                p2,
                p_meas,
                p_prep,
            )
            instances = [(cached_slices.initialization, 0), (cached_slices.terminal, 1)]
        else:
            cached_slices = _cached_surface_memory_dem_slices(
                geometry.dx,
                geometry.dz,
                geometry.orientation.name,
                geometry.rotated,
                basis,
                p1,
                p2,
                p_meas,
                p_prep,
            )
            instances = [(cached_slices.initialization, 0)]
            instances.extend((cached_slices.bulk, round_) for round_ in range(1, operation.rounds - 1))
            instances.extend(
                [
                    (cached_slices.pre_terminal, operation.rounds - 1),
                    (cached_slices.terminal, operation.rounds),
                ],
            )
        schedule = DemSliceRoundSchedule.from_cached_slices(
            cached_slices.output_model,
            instances,
            expected_dem_outputs=self._assembled_dem_output_ids(),
            expected_tracked_paulis=[],
            coordinate_offset=(float(coord_x), float(coord_y)),
            detector_order_routings=self._assembled_detector_order_routings(),
        )
        model = schedule.compose(
            start_round=0,
            commit_rounds=operation.rounds + 1,
            buffer_rounds=0,
            forward_boundary="hard",
        )
        return model, schedule

    def _build_structured_multi_memory_dem_from_cached_slices(
        self,
        *,
        p1: float,
        p2: float,
        p_meas: float,
        p_prep: float,
    ) -> tuple[object, object] | None:
        """Assemble simultaneous independent patch memories from one family."""
        if len(self._patches) < 2 or len(self._operations) != 1:
            return None
        operation = self._operations[0]
        patch_order = list(self._patches)
        if operation.gate_type != LogicalGateType.MEMORY or operation.patches != patch_order or operation.rounds < 1:
            return None

        patch_states = [self._patches[label] for label in patch_order]
        patch_specs = tuple(
            (
                state.patch.geometry.dx,
                state.patch.geometry.dz,
                state.patch.geometry.orientation.name,
                state.patch.geometry.rotated,
            )
            for state in patch_states
        )
        bases = tuple(operation.per_patch_basis.get(label, operation.basis).upper() for label in patch_order)
        singleton = operation.rounds == 1
        cached_slices = _cached_surface_multi_memory_dem_slices(
            patch_specs,
            bases,
            singleton,
            p1,
            p2,
            p_meas,
            p_prep,
        )

        instances = [(cached_slices.initialization, 0)]
        if singleton:
            instances.append((cached_slices.terminal, 1))
        else:
            if cached_slices.bulk is None or cached_slices.pre_terminal is None:  # pragma: no cover - cache invariant
                msg = "multi-memory cache omitted a required non-singleton cached slice"
                raise ValueError(msg)
            instances.extend((cached_slices.bulk, round_) for round_ in range(1, operation.rounds - 1))
            instances.extend(
                [
                    (cached_slices.pre_terminal, operation.rounds - 1),
                    (cached_slices.terminal, operation.rounds),
                ],
            )

        detector_coordinate_offsets = {}
        stream_start = 0
        for state, stream_count, (origin_x, origin_y) in zip(
            patch_states,
            cached_slices.stream_counts,
            cached_slices.coordinate_origins,
            strict=True,
        ):
            patch_x, patch_y = state.coord_offset
            translation = (float(patch_x) - origin_x, float(patch_y) - origin_y)
            detector_coordinate_offsets.update(
                dict.fromkeys(range(stream_start, stream_start + stream_count), translation),
            )
            stream_start += stream_count

        from pecos_rslib.qec import DemSliceRoundSchedule

        schedule = DemSliceRoundSchedule.from_cached_slices(
            cached_slices.output_model,
            instances,
            expected_dem_outputs=self._assembled_dem_output_ids(),
            expected_tracked_paulis=[],
            detector_coordinate_offsets=detector_coordinate_offsets,
            detector_order_routings=self._assembled_detector_order_routings(),
        )
        model = schedule.compose(
            start_round=0,
            commit_rounds=operation.rounds + 1,
            buffer_rounds=0,
            forward_boundary="hard",
        )
        return model, schedule

    def _assemble_boundary_dem_provider(
        self,
        provider: _BoundaryDemProviderDescription,
    ) -> tuple[object, object]:
        """Assemble a provider description already checked by its producer."""
        boundary_slices = list(provider.boundary_slices)
        placements, boundary_round = _boundary_slice_placements(
            boundary_slices,
            list(provider.memory_rounds),
        )
        instances = [(cached_slice, round_) for cached_slice, round_, _ in placements]
        dem_output_routings = _boundary_output_routings(provider, placements)

        from pecos_rslib.qec import DemSliceRoundSchedule

        schedule = DemSliceRoundSchedule.from_cached_slices(
            boundary_slices[0].output_model,
            instances,
            expected_dem_outputs=self._assembled_dem_output_ids(),
            expected_tracked_paulis=[],
            coordinate_offset=provider.coordinate_offset,
            detector_coordinate_offsets=provider.detector_coordinate_offsets,
            dem_output_routings=dem_output_routings,
            detector_order_routings=self._assembled_detector_order_routings(),
        )
        model = schedule.compose(
            start_round=0,
            commit_rounds=boundary_round + 1,
            buffer_rounds=0,
            forward_boundary="hard",
        )
        return model, schedule

    def _build_structured_mixed_dem_from_cached_slices(
        self,
        *,
        p1: float,
        p2: float,
        p_meas: float,
        p_prep: float,
    ) -> tuple[object, object] | None:
        """Assemble two-patch schedules mixing H and CX boundary families."""
        if len(self._patches) != 2 or len(self._operations) < 3 or len(self._operations) % 2 == 0:
            return None

        memories = self._operations[::2]
        gates = self._operations[1::2]
        patch_order = list(self._patches)
        control_label, target_label = patch_order
        if any(
            memory.gate_type != LogicalGateType.MEMORY or memory.rounds < 2 or memory.patches != patch_order
            for memory in memories
        ):
            return None

        gate_names = []
        for gate in gates:
            if gate.gate_type == LogicalGateType.TRANSVERSAL_H and gate.patches == [control_label]:
                gate_names.append("h0")
            elif gate.gate_type == LogicalGateType.TRANSVERSAL_H and gate.patches == [target_label]:
                gate_names.append("h1")
            elif (
                gate.gate_type == LogicalGateType.TRANSVERSAL_CX
                and gate.patches == patch_order
                and not gate.teleportation
                and gate.injection_type is None
            ):
                gate_names.append("cx")
            else:
                return None
        # Pure-CX schedules use the smaller specialized family above.
        if "h0" not in gate_names and "h1" not in gate_names:
            return None

        control_state = self._patches[control_label]
        target_state = self._patches[target_label]
        control_geometry = control_state.patch.geometry
        target_geometry = target_state.patch.geometry
        if "cx" in gate_names and (
            control_geometry.dx != target_geometry.dx
            or control_geometry.dz != target_geometry.dz
            or control_geometry.rotated != target_geometry.rotated
        ):
            return None

        prefix_states = []
        swapped = (False, False)
        for gate_name in gate_names:
            prefix_states.append(swapped)
            if gate_name == "h0":
                swapped = (not swapped[0], swapped[1])
            elif gate_name == "h1":
                swapped = (swapped[0], not swapped[1])
            elif swapped[0] != swapped[1]:
                # A transversal CX does not map the currently assigned surface
                # stabilizers index-for-index when only one patch is H-swapped.
                return None

        initial_control_basis = memories[0].per_patch_basis.get(control_label, memories[0].basis).upper()
        initial_target_basis = memories[0].per_patch_basis.get(target_label, memories[0].basis).upper()
        final_control_basis = memories[-1].per_patch_basis.get(control_label, memories[-1].basis).upper()
        final_target_basis = memories[-1].per_patch_basis.get(target_label, memories[-1].basis).upper()
        # The surface frontend treats an observable as unreliable when any CX
        # partner is finally measured in the incompatible basis. That state is
        # history-sensitive: two CX gates cancel in the sign-free Clifford
        # transform below, but not in the frontend's observable declarations.
        # Until reliability is part of the canonical state, retain the exact
        # full-model fallback for every mixed-basis schedule containing CX.
        if "cx" in gate_names and final_control_basis != final_target_basis:
            return None

        boundary_slices = []
        cached_layout = None
        for boundary_index, selected_gate in enumerate(gate_names):
            start_swapped = prefix_states[boundary_index]
            after_selected = start_swapped
            if selected_gate == "h0":
                after_selected = (not start_swapped[0], start_swapped[1])
            elif selected_gate == "h1":
                after_selected = (start_swapped[0], not start_swapped[1])

            future_transform = _TWO_PATCH_IDENTITY
            future_swapped = after_selected
            for future_gate in gate_names[boundary_index + 1 :]:
                future_transform = _append_two_patch_gate_transform(future_transform, future_gate)
                if future_gate == "h0":
                    future_swapped = (not future_swapped[0], future_swapped[1])
                elif future_gate == "h1":
                    future_swapped = (future_swapped[0], not future_swapped[1])
            future_word = _canonical_two_patch_suffix(after_selected, future_transform, future_swapped)
            fixture_key = (
                control_geometry.dx,
                control_geometry.dz,
                control_geometry.orientation.name,
                control_geometry.rotated,
                target_geometry.dx,
                target_geometry.dz,
                target_geometry.orientation.name,
                target_geometry.rotated,
                initial_control_basis,
                initial_target_basis,
                final_control_basis,
                final_target_basis,
                selected_gate,
                start_swapped,
                future_word,
                p1,
                p2,
                p_meas,
                p_prep,
            )
            cached_layout = _cached_surface_mixed_dem_slices(*fixture_key)
            boundary_slices.append(cached_layout.cached_slices)

        if cached_layout is None:  # Defensive: the alternating form always has at least one gate.
            return None
        detector_coordinate_offsets = _two_patch_detector_coordinate_offsets(
            control_state.coord_offset,
            target_state.coord_offset,
            control_stream_count=cached_layout.control_stream_count,
            target_stream_count=cached_layout.target_stream_count,
            target_coordinate_origin=cached_layout.target_coordinate_origin,
        )
        provider = _BoundaryDemProviderDescription(
            boundary_slices=tuple(boundary_slices),
            memory_rounds=tuple(memory.rounds for memory in memories),
            output_routing=_BoundaryOutputRouting.IDENTITY,
            detector_coordinate_offsets=detector_coordinate_offsets,
        )
        return self._assemble_boundary_dem_provider(provider)

    def _build_structured_cx_dem_from_cached_slices(
        self,
        *,
        p1: float,
        p2: float,
        p_meas: float,
        p_prep: float,
    ) -> tuple[object, object] | None:
        """Assemble alternating memory/CX operations from bounded families."""
        if len(self._patches) != 2 or len(self._operations) < 3 or len(self._operations) % 2 == 0:
            return None

        memories = self._operations[::2]
        gates = self._operations[1::2]
        patch_order = list(self._patches)
        control_label, target_label = patch_order
        if any(
            memory.gate_type != LogicalGateType.MEMORY or memory.rounds < 2 or memory.patches != patch_order
            for memory in memories
        ) or any(
            gate.gate_type != LogicalGateType.TRANSVERSAL_CX
            or gate.patches != patch_order
            or gate.teleportation
            or gate.injection_type is not None
            for gate in gates
        ):
            return None

        control_state = self._patches[control_label]
        target_state = self._patches[target_label]
        control_geometry = control_state.patch.geometry
        target_geometry = target_state.patch.geometry
        if (
            control_geometry.dx != target_geometry.dx
            or control_geometry.dz != target_geometry.dz
            or control_geometry.rotated != target_geometry.rotated
        ):
            return None
        initial_control_basis = memories[0].per_patch_basis.get(control_label, memories[0].basis).upper()
        initial_target_basis = memories[0].per_patch_basis.get(target_label, memories[0].basis).upper()
        final_control_basis = memories[-1].per_patch_basis.get(control_label, memories[-1].basis).upper()
        final_target_basis = memories[-1].per_patch_basis.get(target_label, memories[-1].basis).upper()
        if final_control_basis not in {"X", "Z"} or final_target_basis not in {"X", "Z"}:
            return None
        if len(gates) > 1 and final_control_basis != final_target_basis:
            return None
        fixture_key = (
            control_geometry.dx,
            control_geometry.dz,
            control_geometry.orientation.name,
            control_geometry.rotated,
            target_geometry.dx,
            target_geometry.dz,
            target_geometry.orientation.name,
            target_geometry.rotated,
            initial_control_basis,
            initial_target_basis,
            final_control_basis,
            final_target_basis,
            p1,
            p2,
            p_meas,
            p_prep,
        )
        cached = _cached_surface_cx_dem_slices(*fixture_key)
        cached_slices = cached.cached_slices

        detector_coordinate_offsets = _two_patch_detector_coordinate_offsets(
            control_state.coord_offset,
            target_state.coord_offset,
            control_stream_count=cached.control_stream_count,
            target_stream_count=cached.target_stream_count,
            target_coordinate_origin=cached.target_coordinate_origin,
        )
        provider = _BoundaryDemProviderDescription(
            boundary_slices=(cached_slices,) * len(gates),
            memory_rounds=tuple(memory.rounds for memory in memories),
            output_routing=_BoundaryOutputRouting.REPEATED_CX,
            detector_coordinate_offsets=detector_coordinate_offsets,
            final_basis=final_control_basis,
        )
        return self._assemble_boundary_dem_provider(provider)

    def _build_structured_h_dem_from_cached_slices(
        self,
        *,
        p1: float,
        p2: float,
        p_meas: float,
        p_prep: float,
    ) -> tuple[object, object] | None:
        """Assemble alternating memory/H operations from bounded families."""
        if len(self._patches) != 1 or len(self._operations) < 3 or len(self._operations) % 2 == 0:
            return None

        memories = self._operations[::2]
        gates = self._operations[1::2]
        patch_label = next(iter(self._patches))
        if any(
            memory.gate_type != LogicalGateType.MEMORY or memory.rounds < 2 or memory.patches != [patch_label]
            for memory in memories
        ) or any(gate.gate_type != LogicalGateType.TRANSVERSAL_H or gate.patches != [patch_label] for gate in gates):
            return None

        patch_state = self._patches[patch_label]
        geometry = patch_state.patch.geometry
        initial_basis = memories[0].per_patch_basis.get(patch_label, memories[0].basis).upper()
        final_basis = memories[-1].per_patch_basis.get(patch_label, memories[-1].basis).upper()
        coord_x, coord_y = patch_state.coord_offset
        fixture_keys = [
            (
                geometry.dx,
                geometry.dz,
                geometry.orientation.name,
                geometry.rotated,
                initial_basis,
                final_basis,
                p1,
                p2,
                p_meas,
                p_prep,
                bool(boundary_index % 2),
                bool((len(gates) - boundary_index - 1) % 2),
            )
            for boundary_index in range(len(gates))
        ]
        boundary_slices = tuple(
            _cached_surface_h_dem_slices(
                *key[:-2],
                pre_gate_swapped=key[-2],
                future_h_parity=key[-1],
            )
            for key in fixture_keys
        )
        provider = _BoundaryDemProviderDescription(
            boundary_slices=boundary_slices,
            memory_rounds=tuple(memory.rounds for memory in memories),
            output_routing=_BoundaryOutputRouting.IDENTITY,
            coordinate_offset=(float(coord_x), float(coord_y)),
        )
        return self._assemble_boundary_dem_provider(provider)

    def build_dem(
        self,
        *,
        p1: float = 0.001,
        p2: float = 0.001,
        p_meas: float = 0.001,
        p_prep: float = 0.0,
    ) -> str:
        """Generate a DEM using the PECOS-native fault analysis pipeline.

        TickCircuit -> DagCircuit -> DagFaultAnalyzer -> DemBuilder.
        No Stim dependency. Eligible single-patch memories, repeated
        transversal-H, repeated two-patch transversal-CX, and mixed two-patch
        H/CX algorithms reuse bounded physical fixture compiles across
        requested memory lengths.

        Args:
            p1: Single-qubit depolarizing error rate.
            p2: Two-qubit depolarizing error rate.
            p_meas: Measurement error rate.
            p_prep: Preparation error rate.

        Returns:
            DEM string in Stim-compatible format.
        """
        cached = self._build_structured_dem_from_cached_slices(
            p1=p1,
            p2=p2,
            p_meas=p_meas,
            p_prep=p_prep,
        )
        if cached is None:
            dem, _, _ = self._build_structured_dem(p1=p1, p2=p2, p_meas=p_meas, p_prep=p_prep)
        else:
            dem, _ = cached
        return str(dem)

    def build_sampler_and_decoder(
        self,
        *,
        p1: float = 0.001,
        p2: float = 0.001,
        p_meas: float = 0.001,
        p_prep: float = 0.0,
        inner_decoder: str = "pymatching",
    ) -> tuple[object, object, str]:
        """Build a DemSampler and OSD decoder without any string round-trip.

        Returns:
            Tuple of (DemSampler, LogicalSubgraphDecoder, dem_str).
            dem_str is also returned for compatibility with existing code.
        """
        from pecos_rslib.qec import LogicalSubgraphDecoder

        cached = self._build_structured_dem_from_cached_slices(
            p1=p1,
            p2=p2,
            p_meas=p_meas,
            p_prep=p_prep,
        )
        if cached is None:
            dem, _, _ = self._build_structured_dem(p1=p1, p2=p2, p_meas=p_meas, p_prep=p_prep)
        else:
            dem, _ = cached
        sampler = dem.to_sampler()
        dem_str = str(dem)

        sc = self.stab_coords()
        decoder = LogicalSubgraphDecoder(dem_str, sc, inner_decoder)

        return sampler, decoder, dem_str

    def _z_frame_slot(self, label: str) -> int:
        """Z slot in the descriptor's per-patch (X, Z) frame layout."""
        return list(self._patches).index(label) * 2 + 1

    def build_algorithm_descriptor(
        self,
        *,
        p1: float = 0.001,
        p2: float = 0.001,
        p_meas: float = 0.001,
        p_prep: float = 0.0,
        buffer: int | None = None,
    ) -> dict:
        """Extract per-segment DEMs and boundary gates for LogicalAlgorithmDecoder.

        Splits the structured circuit DEM at gate boundaries. Each memory operation
        becomes a segment; each transversal gate becomes a boundary gate with
        Pauli frame propagation rules. By default, each non-terminal segment
        derives the minimum safe look-ahead from the source-tracked model.
        Passing ``buffer`` requests that exact amount of look-behind and
        look-ahead; a value below the model's required look-ahead is rejected.
        Eligible single-patch memories, repeated H, repeated CX, and mixed
        two-patch H/CX algorithms are assembled directly from bounded
        physical fixture caches; other circuits retain full-model fallback.

        Returns:
            Dict with keys: segments, boundary_gates, num_observables,
            num_frame_slots, full_dem. ``num_observables`` is the full DEM's
            declared observable count; ``num_frame_slots`` is two per patch
            (X then Z). ``injection_readouts`` carries raw ancilla parity
            records separately from deterministic observables and frame slots.
            It is emitted for a future consumer; no decoder applies the
            correction today. T decision-point execution remains unsupported.
        """
        if buffer is not None and buffer < 0:
            msg = "buffer must be non-negative or None"
            raise ValueError(msg)

        # Eligible memory, repeated-H, repeated-CX, and mixed two-patch H/CX
        # algorithms are assembled entirely from bounded physical fixture
        # caches. Other algorithms retain the full structured path as an
        # equivalence oracle and fallback until their logical-operation families
        # are available.
        cached = self._build_structured_dem_from_cached_slices(
            p1=p1,
            p2=p2,
            p_meas=p_meas,
            p_prep=p_prep,
        )
        if cached is None:
            structured_dem, influence_map, dag_circuit = self._build_structured_dem(
                p1=p1,
                p2=p2,
                p_meas=p_meas,
                p_prep=p_prep,
            )
            round_schedule = structured_dem.round_schedule(influence_map, dag_circuit)
        else:
            structured_dem, round_schedule = cached
        full_dem = str(structured_dem)
        sc = self.stab_coords()

        # Compute segment time boundaries from operations.
        # Each MEMORY op has a number of rounds. Time coordinates are
        # sequential round indices across all segments.
        segments = []
        boundary_gates = []
        # Gates accumulate between consecutive MEMORY ops.
        pending_gates = []
        time_cursor = 0
        patch_labels = list(self._patches.keys())
        num_patches = len(patch_labels)

        # Track X/Z swap state per patch for stab_coords.
        # After transversal H, the X and Z stabilizer types swap.
        x_z_swapped = dict.fromkeys(patch_labels, False)

        for op in self._operations:
            if op.gate_type == LogicalGateType.MEMORY:
                # If there are pending gates, they form the boundary
                # between the previous segment and this one.
                if segments and pending_gates:
                    boundary_gates.append(pending_gates)
                    pending_gates = []
                elif segments:
                    # No gate between segments — empty boundary
                    boundary_gates.append([])
                    pending_gates = []
                seg_start = time_cursor
                seg_end = time_cursor + op.rounds
                time_cursor = seg_end

                # Build per-segment stab_coords respecting X/Z swap state
                seg_sc = []
                for label in patch_labels:
                    base = sc[patch_labels.index(label)]
                    if x_z_swapped[label]:
                        # Swap X and Z positions
                        seg_sc.append({"X": base["Z"], "Z": base["X"]})
                    else:
                        seg_sc.append({"X": base["X"], "Z": base["Z"]})

                segments.append(
                    {
                        "time_start": seg_start,
                        "time_end": seg_end,
                        "stab_coords": seg_sc,
                    },
                )

            elif op.gate_type == LogicalGateType.TRANSVERSAL_H:
                label = op.patches[0]
                idx = patch_labels.index(label)
                pending_gates.append(
                    {
                        "type": "Hadamard",
                        "x_obs_bit": idx * 2,
                        "z_obs_bit": self._z_frame_slot(label),
                    },
                )
                x_z_swapped[label] = not x_z_swapped[label]

            elif op.gate_type == LogicalGateType.TRANSVERSAL_CX:
                ctrl_label, tgt_label = op.patches[0], op.patches[1]
                ctrl_idx = patch_labels.index(ctrl_label)
                tgt_idx = patch_labels.index(tgt_label)
                if op.injection_type == "T":
                    pending_gates.append(
                        {
                            "type": "TGateInjection",
                            "z_obs_bit": self._z_frame_slot(ctrl_label),
                            "ancilla_z_bit": self._z_frame_slot(tgt_label),
                        },
                    )
                else:
                    pending_gates.append(
                        {
                            "type": "Cnot",
                            "ctrl_x_bit": ctrl_idx * 2,
                            "ctrl_z_bit": self._z_frame_slot(ctrl_label),
                            "tgt_x_bit": tgt_idx * 2,
                            "tgt_z_bit": self._z_frame_slot(tgt_label),
                        },
                    )

            elif op.gate_type in (LogicalGateType.TRANSVERSAL_SZ, LogicalGateType.TRANSVERSAL_SZdg):
                label = op.patches[0]
                idx = patch_labels.index(label)
                pending_gates.append(
                    {
                        "type": "SGate",
                        "x_obs_bit": idx * 2,
                        "z_obs_bit": self._z_frame_slot(label),
                    },
                )

        if not segments:
            msg = "algorithm descriptor must contain at least one segment"
            raise ValueError(msg)

        _validate_boundary_cardinality(segments, boundary_gates)

        # Assemble each sub-DEM through the native slice schedule. This keeps
        # independent source contributions, decomposition metadata, logical
        # outputs, hyperedges, and cross-round correlations intact.
        seg_dems = []
        segment_detector_counts = []
        segment_window_detector_counts = []
        detector_rounds = []
        for detector_id, coords in structured_dem.detector_coordinates():
            if coords is None or len(coords) < 3:
                msg = (
                    f"detector {detector_id} has no [x, y, round] coordinates; "
                    "algorithm segmentation requires an explicit round for every detector"
                )
                raise ValueError(msg)
            detector_rounds.append(int(coords[2]))
        explicit_buffer = 0 if buffer is None else buffer
        for segment_index, seg in enumerate(segments):
            start_round = max(0, int(seg["time_start"]) - explicit_buffer)
            is_last = segment_index == len(segments) - 1
            commit_end = time_cursor + 1 if is_last else int(seg["time_end"])
            commit_rounds = commit_end - start_round
            forward_buffer = 0 if is_last else buffer
            forward_boundary = "hard" if is_last else "soft"

            if not is_last and buffer is not None:
                required = round_schedule.required_buffer_rounds(
                    start_round,
                    commit_rounds,
                )
                if buffer < required:
                    msg = (
                        f"buffer={buffer} is too small for logical segment {segment_index}; "
                        f"the source-tracked DEM requires at least {required} look-ahead rounds"
                    )
                    raise ValueError(msg)

            segment_dem = round_schedule.compose(
                start_round=start_round,
                commit_rounds=commit_rounds,
                buffer_rounds=forward_buffer,
                forward_boundary=forward_boundary,
            )
            seg_dems.append(str(segment_dem))
            segment_window_detector_counts.append(segment_dem.num_detectors)
            # Segment metadata partitions the incoming full-circuit syndrome;
            # it therefore counts only this segment's commit detectors, not the
            # look-behind/look-ahead detectors duplicated in its local DEM.
            segment_detector_counts.append(
                sum(int(seg["time_start"]) <= round_ < commit_end for round_ in detector_rounds),
            )

        # Physical code distance for latency/windowing decisions. With multiple
        # patches use the minimum (the weakest bound governs latency). This is the
        # real surface-code distance, NOT a count of logical patches.
        distance = min(
            (min(ps.patch.geometry.dx, ps.patch.geometry.dz) for ps in self._patches.values()),
            default=0,
        )

        from pecos_rslib.qec import ParsedDem

        num_observables = ParsedDem.from_string(full_dem).num_observables
        injection_readouts = json.loads(self.to_tick_circuit().get_meta("injection_readouts"))
        for readout in injection_readouts:
            readout["data_z_frame_slot"] = self._z_frame_slot(readout["data_patch"])
            readout["ancilla_z_frame_slot"] = self._z_frame_slot(readout["ancilla_patch"])

        return {
            "segments": [
                {
                    "dem": seg_dems[i],
                    "num_detectors": segment_detector_counts[i],
                    "num_commit_detectors": segment_detector_counts[i],
                    "num_window_detectors": segment_window_detector_counts[i],
                    "stab_coords": segments[i]["stab_coords"],
                }
                for i in range(len(segments))
            ],
            "boundary_gates": boundary_gates,
            "num_observables": num_observables,
            "num_frame_slots": num_patches * 2,
            "injection_readouts": injection_readouts,
            "full_dem": full_dem,
            "distance": distance,
        }

    def build_decoder(
        self,
        *,
        p1: float = 0.001,
        p2: float = 0.001,
        p_meas: float = 0.001,
        p_prep: float = 0.0,
        inner_decoder: str = "fusion_blossom_serial",
        use_stim_dem: bool = True,
    ) -> tuple[object, object]:
        """Build an LogicalSubgraphDecoder for this circuit.

        Args:
            p1: Single-qubit depolarizing error rate.
            p2: Two-qubit depolarizing error rate.
            p_meas: Measurement error rate.
            p_prep: Preparation error rate.
            inner_decoder: Decoder type for each subgraph.
            use_stim_dem: If True, use Stim for DEM generation (more error
                mechanisms). If False, use PECOS-native DEM pipeline.

        Returns:
            Tuple of (stim.Circuit, LogicalSubgraphDecoder).
        """
        import stim
        from pecos_rslib.qec import LogicalSubgraphDecoder

        stim_str = self.to_stim(p1=p1, p2=p2, p_meas=p_meas, p_prep=p_prep)
        circuit = stim.Circuit(stim_str)

        if use_stim_dem:
            dem = circuit.detector_error_model(ignore_decomposition_failures=True)
            dem_str = str(dem)
        else:
            dem_str = self.build_dem(p1=p1, p2=p2, p_meas=p_meas, p_prep=p_prep)

        sc = self.stab_coords()
        decoder = LogicalSubgraphDecoder(dem_str, sc, inner_decoder)
        return circuit, decoder


class _CircuitGenerator:
    """Internal: generates a PECOS TickCircuit for logical circuits.

    Builds a TickCircuit with detector and observable annotations as
    JSON metadata. The TickCircuit is the source of truth; Stim circuit
    strings are derived from it via tick_circuit_to_stim().
    """

    def __init__(
        self,
        patches: dict[str, PatchState],
        operations: list[LogicalOp],
    ) -> None:
        from pecos_rslib.quantum import TickCircuit

        self.patches = patches
        self.operations = operations
        self.tc = TickCircuit()
        self._current_tick = None
        self._allocated: set[int] = set()
        self.meas_count = 0

        self.stab_meas: dict[tuple[str, str, int, int, int], int] = {}
        self._stab_meas_by_round: dict[tuple[str, int, int], list[tuple[str, str, int, int, int]]] = {}
        self._last_round: dict[tuple[str, str, int], int] = {}
        self._boundary_terms: dict[tuple[str, str, int], list[tuple[str, str, int]] | None] = {}
        self._propagation_context = _PropagationContext.from_operations(operations)
        self.data_meas: dict[tuple[str, int], int] = {}
        self._injection_ops = [op for op in operations if op.teleportation]
        self._injection_ancillas = {op.patches[1] for op in self._injection_ops}
        self._injection_readouts: dict[str, dict] = {}

        self._prepared: set[str] = set()
        self.segment_idx = 0
        self.next_observable_idx = 0
        self.round_time = 0.0

        self._det_json: list[dict] = []
        self._obs_json: list[dict] = []

    def _new_tick(self) -> object:
        self._current_tick = self.tc.tick()
        return self._current_tick

    def _tick(self) -> object:
        if self._current_tick is None:
            return self._new_tick()
        return self._current_tick

    def _end_tick(self) -> None:
        if self._current_tick is not None:
            round_owner = int(self.round_time)
            if float(round_owner) != self.round_time:
                msg = f"DEM slice round owner must be integral, got {self.round_time}"
                raise ValueError(msg)
            tick_idx = self._current_tick.index()
            tick = self.tc.get_tick(tick_idx)
            if tick is not None:
                for gate_idx in range(tick.gate_batch_count()):
                    self.tc.set_gate_meta(
                        tick_idx,
                        gate_idx,
                        DEM_SLICE_ROUND_ATTRIBUTE,
                        round_owner,
                    )
        self._current_tick = None

    def _emit_qalloc_or_reset(self, qubits: list[int]) -> None:
        t = self._tick()
        new_qs = [q for q in qubits if q not in self._allocated]
        old_qs = [q for q in qubits if q in self._allocated]
        if new_qs:
            t.qalloc(new_qs)
            self._allocated.update(new_qs)
        if old_qs:
            t.pz(old_qs)

    def generate(self) -> object:
        """Generate the TickCircuit with detector/observable metadata."""
        # Per-patch last memory index: for each patch, the last MEMORY
        # operation that includes it. This ensures each patch gets its
        # final measurement emitted in the correct segment.
        last_mem_for_patch: dict[str, int] = {}
        for i, op in enumerate(self.operations):
            if op.gate_type == LogicalGateType.MEMORY:
                for label in op.patches:
                    last_mem_for_patch[label] = i

        for op_idx, op in enumerate(self.operations):
            if op.gate_type == LogicalGateType.MEMORY:
                # A patch is "last" in this segment if this is its last memory op.
                last_patches = {label for label in op.patches if last_mem_for_patch.get(label) == op_idx}
                self._emit_memory_segment(
                    op,
                    is_last=bool(last_patches),
                    last_patches=last_patches,
                )
                self.segment_idx += 1

            elif op.gate_type == LogicalGateType.TRANSVERSAL_H:
                self._emit_transversal_h(op)

            elif op.gate_type == LogicalGateType.TRANSVERSAL_SZ:
                self._emit_transversal_sz(op)

            elif op.gate_type == LogicalGateType.TRANSVERSAL_SZdg:
                self._emit_transversal_szdg(op)

            elif op.gate_type == LogicalGateType.TRANSVERSAL_CX:
                self._emit_transversal_cx(op)

        # Build detector/observable definitions with both formats:
        # - "records": negative offsets (Stim compatibility, legacy)
        # - "meas_ids": absolute MeasResult IDs (stable, preferred)
        total = self.meas_count
        det_out = [
            {
                "id": d["id"],
                "coords": d["coords"],
                "records": [idx - total for idx in d["abs_records"]],
                "meas_ids": d["abs_records"],
            }
            for d in self._det_json
        ]
        obs_out = [
            {
                "id": o["id"],
                "records": [idx - total for idx in o["abs_records"]],
                "meas_ids": o["abs_records"],
            }
            for o in self._obs_json
        ]

        for label in self._injection_ancillas:
            if label not in self._injection_readouts:
                msg = f"Injection ancilla '{label}' has no logical operator for its readout"
                raise ValueError(msg)

        self.tc.set_meta("detectors", json.dumps(det_out))
        self.tc.set_meta("observables", json.dumps(obs_out))
        self.tc.set_meta("num_measurements", str(total))
        # Semantic provenance of every measurement ordinal, for cross-form
        # comparison against Guppy sidebands without reaching into the generator.
        self.tc.set_meta(
            "measurement_keys",
            json.dumps(
                {
                    "stabilizer": [[*key, ordinal] for key, ordinal in self.stab_meas.items()],
                    "data": [[label, qubit, ordinal] for (label, qubit), ordinal in self.data_meas.items()],
                },
            ),
        )
        self.tc.set_meta(
            "injection_readouts",
            json.dumps(
                [
                    {
                        "data_patch": op.patches[0],
                        "ancilla_patch": op.patches[1],
                        "injection_type": op.injection_type,
                        **self._injection_readouts[op.patches[1]],
                        "records": [idx - total for idx in self._injection_readouts[op.patches[1]]["meas_ids"]],
                    }
                    for op in self._injection_ops
                ],
            ),
        )
        if {key for keys in self._stab_meas_by_round.values() for key in keys} != set(self.stab_meas):
            msg = "Round measurement index does not cover the recorded stabilizer measurements"
            raise AssertionError(msg)
        return self.tc

    def _last_memory_basis(self, patch_label: str | None = None) -> str:
        """Basis of this patch's last memory segment (readout)."""
        for op in reversed(self.operations):
            if op.gate_type == LogicalGateType.MEMORY and (patch_label is None or patch_label in op.patches):
                return op.per_patch_basis.get(patch_label, op.basis)
        return "Z"

    def _emit_meas(self, qubits: list[int]) -> list[int]:
        self._tick().mz(qubits)
        indices = list(range(self.meas_count, self.meas_count + len(qubits)))
        self.meas_count += len(qubits)
        return indices

    def _allocation(self, patch_label: str) -> QubitAllocation:
        ps = self.patches[patch_label]
        allocation = gadgets.default_allocation(ps.patch)
        return QubitAllocation(
            [q + ps.qubit_offset for q in allocation.data_qubits],
            [q + ps.qubit_offset for q in allocation.x_ancilla_qubits],
            [q + ps.qubit_offset for q in allocation.z_ancilla_qubits],
        )

    def _emit_steps(self, step_lists: list[tuple[SurfaceCircuitStep, ...]]) -> dict[tuple[int, int], int]:
        """Split different-type conflicts within TICK groups, then zip with padding.

        Each TICK-delimited group, including a comment-only group, produces
        at least one tick; trailing empty groups at the end of the merged
        stream are omitted. Physical steps must contain at least one qubit.

        Patch and step order determine measurement indices. Ancillas retain
        allocation across rounds, unlike TickCircuitRenderer's freeing readout.
        Repeated same-type operations on a qubit in one group are invalid
        parallel layers and must not be silently serialized.
        TickCircuitRenderer retains its silent same-type split until the
        non-rotated schedule is fixed.
        """
        streams = []
        for position, steps in enumerate(step_lists):
            groups = []
            subticks = []
            current = []
            used = set()
            group_used = set()
            layer_label = "unnamed"
            for step in steps:
                if step.op_type == OpType.COMMENT:
                    layer_label = step.label
                    continue
                if step.op_type == OpType.TICK:
                    if current:
                        subticks.append(current)
                    groups.append(subticks or [[]])
                    subticks, current, used = [], [], set()
                    group_used = set()
                    continue
                layer = f"Gadget layer {position}:{len(groups)} ({layer_label})"
                if not step.qubits:
                    msg = f"{layer}: {step.op_type.name} requires at least one qubit"
                    raise ValueError(msg)
                if step.op_type == OpType.MEASURE and len(step.qubits) != 1:
                    msg = f"{layer}: MEASURE requires exactly one qubit per labeled step"
                    raise ValueError(msg)
                for qubit in step.qubits:
                    key = (step.op_type, qubit)
                    if key in group_used:
                        msg = f"{layer}: repeated {step.op_type.name} on qubit {qubit} in a parallel layer"
                        raise ValueError(msg)
                    group_used.add(key)
                if used.intersection(step.qubits):
                    subticks.append(current)
                    current, used = [], set()
                current.append(step)
                used.update(step.qubits)
            if current:
                subticks.append(current)
            if subticks:
                groups.append(subticks)
            streams.append(groups)

        measurements = {}
        merged_groups = list(zip_longest(*streams, fillvalue=()))
        while merged_groups and not any(step for group in merged_groups[-1] for subtick in group for step in subtick):
            merged_groups.pop()
        for groups in merged_groups:
            for subticks in zip_longest(*groups, fillvalue=()):
                merged = [(position, step) for position, steps in enumerate(subticks) for step in steps]
                t = self._new_tick()
                batches = {}
                for position, step in merged:
                    batches.setdefault(step.op_type, []).append((position, step))
                for op_type, batch in batches.items():
                    qubits = [q for _, step in batch for q in step.qubits]
                    if op_type == OpType.ALLOC:
                        # Preserve patch order when fresh allocations and resets
                        # share a tick. TickCircuit coalesces adjacent gate calls.
                        for _, step in batch:
                            self._emit_qalloc_or_reset(step.qubits)
                    elif op_type == OpType.CX:
                        t.cx([tuple(step.qubits) for _, step in batch])
                    elif op_type in {OpType.H, OpType.SZ, OpType.SZDG, OpType.X, OpType.Z}:
                        getattr(t, op_type.name.lower())(qubits)
                    elif op_type == OpType.MEASURE:
                        indices = iter(self._emit_meas(qubits))
                        for position, step in batch:
                            measurements[position, step.qubits[0]] = next(indices)
                    else:
                        msg = f"Unsupported gadget operation: {op_type.name}"
                        raise NotImplementedError(msg)
                self._end_tick()
        return measurements

    def _emit_memory_segment(
        self,
        op: LogicalOp,
        *,
        is_last: bool,
        last_patches: set[str] | None = None,
    ) -> None:
        """Compose preparation and orientation-aware rounds for each patch."""
        first_patches = set(op.patches) - self._prepared
        allocations = {label: self._allocation(label) for label in op.patches}
        self._emit_steps(
            [
                (
                    gadgets.prep_gadget(
                        self.patches[label].patch,
                        allocations[label],
                        basis=op.per_patch_basis.get(label, op.basis),
                    ).steps
                    if label in first_patches
                    else ()
                )
                for label in op.patches
            ],
        )
        self._prepared.update(first_patches)
        for rnd in range(op.rounds):
            measurements = self._emit_steps(
                [
                    gadgets.syndrome_round_gadget(
                        self.patches[label].patch,
                        allocations[label],
                        round_index=rnd,
                        x_z_swapped=self.patches[label].x_z_swapped,
                    ).steps
                    for label in op.patches
                ],
            )
            for (position, qubit), index in measurements.items():
                label = op.patches[position]
                allocation = allocations[label]
                if qubit in allocation.x_ancilla_qubits:
                    family, register = "X", allocation.x_ancilla_qubits
                elif qubit in allocation.z_ancilla_qubits:
                    family, register = "Z", allocation.z_ancilla_qubits
                else:
                    msg = f"Measured qubit {qubit} is not an ancilla of patch '{label}'"
                    raise ValueError(msg)
                if self.patches[label].x_z_swapped:
                    family = "Z" if family == "X" else "X"
                key = (label, family, register.index(qubit), self.segment_idx, rnd)
                self.stab_meas[key] = index
                self._last_round[label, family, self.segment_idx] = rnd
                self._stab_meas_by_round.setdefault((label, self.segment_idx, rnd), []).append(key)
            for label in op.patches:
                self._emit_round_detectors(label, rnd)
            self.round_time += 1.0

        # Read every final patch before constructing cross-patch observables.
        if is_last and last_patches:
            final_patches = [label for label in op.patches if label in last_patches]
            for label in final_patches:
                self._emit_final_data_measurements(label)
            for label in final_patches:
                self._emit_final_detectors_and_observables(label)

    def _emit_round_detectors(
        self,
        patch_label: str,
        round_idx: int,
    ) -> None:
        """Emit detectors for one syndrome round.

        Round 0 of a segment propagates each check backwards to zero, one, or
        more earlier records across patches (the patch's own preparation closes
        a matching-type check into a singleton) or emits no detector; later
        rounds compare same-type measurements in consecutive rounds.
        """
        seg = self.segment_idx
        # A distance-one patch has no ancillas and records no syndrome keys.
        if self.patches[patch_label].patch.geometry.num_ancilla == 0:
            return

        # Recorded keys carry the current family and the physical register index,
        # including unequal family sizes after H on even-distance patches.
        for curr_key in self._stab_meas_by_round[patch_label, seg, round_idx]:
            _, stab_type, stab_index, _, _ = curr_key
            curr_idx = self.stab_meas[curr_key]

            if round_idx == 0:
                self._emit_boundary_detector(patch_label, stab_type, stab_index, curr_idx)

            elif round_idx > 0:
                # Normal: compare with previous round in same segment
                prev_key = (patch_label, stab_type, stab_index, seg, round_idx - 1)
                prev_idx = self.stab_meas.get(prev_key)
                if prev_idx is not None:
                    self._add_detector(
                        patch_label,
                        stab_type,
                        stab_index,
                        [curr_idx, prev_idx],
                    )

    def _emit_boundary_detector(
        self,
        patch_label: str,
        stab_type: str,
        stab_index: int,
        curr_meas_idx: int,
    ) -> None:
        """Compare the current check with its backwards-propagated measurements."""
        base_family = stab_type
        if self.patches[patch_label].x_z_swapped:
            base_family = "Z" if stab_type == "X" else "X"
        cache_key = (patch_label, base_family, self.segment_idx)
        if cache_key not in self._boundary_terms:
            self._boundary_terms[cache_key] = _propagate_stabilizer_terms(
                self._propagation_context,
                self.segment_idx,
                (patch_label, base_family, stab_type),
            )
        earlier_keys = self._boundary_terms[cache_key]
        if earlier_keys is None:
            return
        records = [curr_meas_idx]
        for label, family, segment in earlier_keys:
            last_round = self._last_round_of_segment(label, family, segment)
            if last_round is None:
                msg = f"no {family} measurement recorded for patch {label} at segment {segment}"
                raise ValueError(msg)
            key = (label, family, stab_index, segment, last_round)
            if key not in self.stab_meas:
                msg = f"index {stab_index} not recorded for patch {label}, family {family}, segment {segment}"
                raise ValueError(msg)
            records.append(self.stab_meas[key])
        self._add_detector(patch_label, stab_type, stab_index, records)

    def _last_round_of_segment(self, patch_label: str, stab_type: str, seg_idx: int) -> int | None:
        """Look up the last recorded round for a stabilizer family in O(1)."""
        return self._last_round.get((patch_label, stab_type, seg_idx))

    def _ancilla_spatial_coords(
        self,
        patch_label: str,
        stab_type: str,
        stab_index: int,
    ) -> tuple[float, float]:
        """Compute the spatial position of a stabilizer's ancilla.

        Returns (x, y) including the patch's coord_offset, using the
        average position of the stabilizer's data qubits.
        """
        ps = self.patches[patch_label]
        geom = ps.patch.geometry
        cx, cy = ps.coord_offset
        base_family = ("Z" if stab_type == "X" else "X") if ps.x_z_swapped else stab_type
        s = ps.stabilizers_by_index[base_family].get(stab_index)
        if s is None:
            msg = f"Patch '{patch_label}' has no {base_family} stabilizer with index {stab_index}"
            raise ValueError(msg)
        positions = [geom.id_to_pos[q] for q in s.data_qubits]
        avg_row = sum(r for r, c in positions) / len(positions)
        avg_col = sum(c for r, c in positions) / len(positions)
        return (avg_col * 2 + cx, avg_row * 2 + cy)

    def _add_detector(
        self,
        patch_label: str,
        stab_type: str,
        stab_index: int,
        meas_indices: list[int],
    ) -> None:
        anc_x, anc_y = self._ancilla_spatial_coords(patch_label, stab_type, stab_index)
        # Store absolute indices; convert to relative offsets in generate()
        self._det_json.append(
            {
                "id": len(self._det_json),
                "coords": [anc_x, anc_y, self.round_time],
                "abs_records": list(meas_indices),
            },
        )

    def _emit_transversal_layer(self, op: LogicalOp, gate: str) -> None:
        label = op.patches[0]
        self._emit_steps(
            [
                gadgets.transversal_layer_gadget(self.patches[label].patch, self._allocation(label), gate=gate).steps,
            ],
        )

    def _emit_transversal_h(self, op: LogicalOp) -> None:
        self._emit_transversal_layer(op, "H")
        ps = self.patches[op.patches[0]]
        ps.x_z_swapped = not ps.x_z_swapped

    def _emit_transversal_sz(self, op: LogicalOp) -> None:
        self._emit_transversal_layer(op, "SZ")

    def _emit_transversal_szdg(self, op: LogicalOp) -> None:
        self._emit_transversal_layer(op, "SZDG")

    def _emit_transversal_cx(self, op: LogicalOp) -> None:
        ctrl_label, tgt_label = op.patches[0], op.patches[1]
        ctrl_ps = self.patches[ctrl_label]
        tgt_ps = self.patches[tgt_label]

        if ctrl_ps.x_z_swapped != tgt_ps.x_z_swapped:
            msg = (
                f"Transversal CX requires same stabilizer orientation. "
                f"'{ctrl_label}' swapped={ctrl_ps.x_z_swapped}, "
                f"'{tgt_label}' swapped={tgt_ps.x_z_swapped}."
            )
            raise ValueError(msg)

        self._emit_steps(
            [
                gadgets.transversal_cx_gadget(
                    ctrl_ps.patch,
                    self._allocation(ctrl_label),
                    tgt_ps.patch,
                    self._allocation(tgt_label),
                ).steps,
            ],
        )

    def _emit_final_data_measurements(self, patch_label: str) -> None:
        ps = self.patches[patch_label]
        measured = self._emit_steps(
            [
                gadgets.measure_out_gadget(
                    ps.patch,
                    self._allocation(patch_label),
                    basis=self._last_memory_basis(patch_label),
                ).steps,
            ],
        )
        for q in range(ps.patch.geometry.num_data):
            self.data_meas[patch_label, q] = measured[0, ps.qubit_offset + q]

    def _emit_final_detectors_and_observables(self, patch_label: str) -> None:
        ps = self.patches[patch_label]
        geom = ps.patch.geometry
        meas_basis = self._last_memory_basis(patch_label)

        if meas_basis == "Z":
            final_stabs = geom.x_stabilizers if ps.x_z_swapped else geom.z_stabilizers
            lookup_type = "Z"
        else:
            final_stabs = geom.z_stabilizers if ps.x_z_swapped else geom.x_stabilizers
            lookup_type = "X"

        if ps.x_z_swapped:
            logical_op = geom.logical_z if meas_basis == "X" else geom.logical_x
        else:
            logical_op = geom.logical_x if meas_basis == "X" else geom.logical_z

        if logical_op is None:
            role = "Injection ancilla" if patch_label in self._injection_ancillas else "Patch"
            msg = f"{role} '{patch_label}' has no logical operator for {meas_basis} readout"
            raise ValueError(msg)

        seg = self.segment_idx
        last_rnd = self._last_round_of_segment(patch_label, lookup_type, seg)

        if last_rnd is not None:
            for s in final_stabs:
                data_rec = [self.data_meas[(patch_label, dq)] for dq in s.data_qubits]
                syn_key = (patch_label, lookup_type, s.index, seg, last_rnd)
                syn_idx = self.stab_meas.get(syn_key)
                if syn_idx is not None:
                    all_idx = [*data_rec, syn_idx]
                    anc_x, anc_y = self._ancilla_spatial_coords(patch_label, lookup_type, s.index)
                    self._det_json.append(
                        {
                            "id": len(self._det_json),
                            "coords": [anc_x, anc_y, self.round_time],
                            "abs_records": list(all_idx),
                        },
                    )

        obs_indices = [self.data_meas[(patch_label, q)] for q in logical_op.data_qubits]
        if patch_label in self._injection_ancillas:
            # A consumed ancilla's random logical readout controls a
            # correction; it is not a deterministic DEM observable.
            self._injection_readouts[patch_label] = {"basis": meas_basis, "meas_ids": obs_indices}
            return
        if not _logical_readout_is_deterministic(self.operations, self.segment_idx, patch_label, meas_basis):
            # Skip non-reliable observables — they're physically
            # non-deterministic and would cause Stim DEM errors.
            self.next_observable_idx += 1
            return

        obs_idx = self.next_observable_idx
        self.next_observable_idx += 1
        self._obs_json.append(
            {
                "id": obs_idx,
                "abs_records": list(obs_indices),
            },
        )
