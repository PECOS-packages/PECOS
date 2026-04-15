# Copyright 2024 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Surface code patch with runtime configuration.

Provides a flexible, runtime-configurable surface code patch
with geometry stored as data structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, TypedDict

from pecos_rslib.num import zeros

from pecos.qec.surface.layouts import (
    compute_rotated_x_stabilizers,
    compute_rotated_z_stabilizers,
    compute_x_stabilizer_supports,
    compute_z_stabilizer_supports,
    get_rotated_logical_x,
    get_rotated_logical_z,
)
from pecos.qec.surface.schedule import get_stab_schedule

if TYPE_CHECKING:
    import pecos


class PatchOrientation(Enum):
    """Orientation of the surface code patch boundaries."""

    X_TOP_BOTTOM = auto()  # X boundaries on top/bottom, Z on left/right
    Z_TOP_BOTTOM = auto()  # Z boundaries on top/bottom, X on left/right


@dataclass(frozen=True)
class Stabilizer:
    """A stabilizer measurement in the surface code."""

    index: int
    stab_type: str  # 'X' or 'Z'
    data_qubits: tuple[int, ...]
    is_boundary: bool
    position: tuple[int, int] = (0, 0)

    @property
    def weight(self) -> int:
        """Number of data qubits in this stabilizer."""
        return len(self.data_qubits)


@dataclass(frozen=True)
class LogicalOperator:
    """A logical operator for the surface code."""

    op_type: str
    data_qubits: tuple[int, ...]


class StabilizerScheduleEntry(TypedDict):
    """Public metadata for one stabilizer schedule touch."""

    round_0based: int
    data_qubit: int
    touch_label: str


class SurfacePatchDescriptor(TypedDict):
    """Public summary of one surface-code patch."""

    distance: int
    dx: int
    dz: int
    rotated: bool
    orientation: str
    num_data: int
    num_ancilla: int
    num_qubits: int


class StabilizerDescriptor(TypedDict):
    """Public descriptor for one stabilizer."""

    stabilizer_kind: str
    stabilizer_index: int
    stabilizer_is_boundary: bool
    stabilizer_region: str
    schedule_rounds: list[int]
    schedule_start_round: int | None
    schedule_end_round: int | None
    schedule_entries: list[StabilizerScheduleEntry]
    data_qubits: list[int]
    data_qubit_positions: list[list[int]]
    weight: int


class LogicalDescriptor(TypedDict):
    """Public descriptor for one logical operator."""

    logical_type: str
    data_qubits: list[int]
    data_qubit_positions: list[list[int]]
    weight: int
    support_axis: str


def _get_stabilizer_region(stab: Stabilizer, patch: SurfacePatch) -> str:
    """Return a coarse region label like ``top+left`` for a stabilizer."""
    geom = patch.geometry
    positions = [geom.id_to_pos[q] for q in stab.data_qubits]
    avg_row = sum(row for row, _ in positions) / len(positions)
    avg_col = sum(col for _, col in positions) / len(positions)
    row_label = "top" if avg_row < (geom.dx - 1) / 2 else "bottom"
    col_label = "left" if avg_col < (geom.dz - 1) / 2 else "right"
    return f"{row_label}+{col_label}"


def _get_stabilizer_touch_label(stab: Stabilizer, patch: SurfacePatch, data_qubit: int) -> str:
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


def _get_stabilizer_schedule_metadata(stab: Stabilizer, patch: SurfacePatch) -> dict[str, object]:
    """Return metadata describing one stabilizer's schedule and geometry."""
    entries: list[StabilizerScheduleEntry] = [
        {
            "round_0based": round_0based,
            "data_qubit": data_qubit,
            "touch_label": _get_stabilizer_touch_label(stab, patch, data_qubit),
        }
        for round_0based, data_qubit in get_stab_schedule(
            stab.stab_type,
            stab.data_qubits,
            stab.is_boundary,
            patch.distance,
        )
    ]
    rounds = [int(entry["round_0based"]) for entry in entries]
    return {
        "stabilizer_kind": stab.stab_type,
        "stabilizer_index": stab.index,
        "stabilizer_is_boundary": stab.is_boundary,
        "stabilizer_region": _get_stabilizer_region(stab, patch),
        "schedule_rounds": rounds,
        "schedule_start_round": rounds[0] if rounds else None,
        "schedule_end_round": rounds[-1] if rounds else None,
        "schedule_entries": entries,
    }


@dataclass
class PatchGeometry:
    """Geometry of a surface code patch.

    Supports both rotated (default) and standard (non-rotated) surface codes.
    The rotated layout is more common and uses fewer qubits for the same distance.
    """

    dx: int
    dz: int
    orientation: PatchOrientation = PatchOrientation.X_TOP_BOTTOM
    rotated: bool = True

    num_data: int = field(init=False)
    num_x_stab: int = field(init=False)
    num_z_stab: int = field(init=False)

    pos_to_id: dict[tuple[int, int], int] = field(default_factory=dict)
    id_to_pos: dict[int, tuple[int, int]] = field(default_factory=dict)

    x_stabilizers: list[Stabilizer] = field(default_factory=list)
    z_stabilizers: list[Stabilizer] = field(default_factory=list)

    logical_x: LogicalOperator | None = None
    logical_z: LogicalOperator | None = None

    def __post_init__(self) -> None:
        """Initialize computed fields and generate geometry."""
        self.num_data = self.dx * self.dz
        self.num_x_stab = (self.dx * self.dz - 1) // 2
        self.num_z_stab = (self.dx * self.dz - 1) // 2

        self._generate_layout()
        self._generate_stabilizers()
        self._generate_logical_operators()

    def _generate_layout(self) -> None:
        for row in range(self.dx):
            for col in range(self.dz):
                idx = row * self.dz + col
                pos = (row, col)
                self.pos_to_id[pos] = idx
                self.id_to_pos[idx] = pos

    def _generate_stabilizers(self) -> None:
        d = min(self.dx, self.dz)

        if self.rotated:
            x_supports = compute_rotated_x_stabilizers(d)
            z_supports = compute_rotated_z_stabilizers(d)
        else:
            x_supports = compute_x_stabilizer_supports(d)
            z_supports = compute_z_stabilizer_supports(d)

        self.x_stabilizers = [
            Stabilizer(
                index=s.index,
                stab_type="X",
                data_qubits=s.data_qubits,
                is_boundary=s.is_boundary,
            )
            for s in x_supports
        ]

        self.z_stabilizers = [
            Stabilizer(
                index=s.index,
                stab_type="Z",
                data_qubits=s.data_qubits,
                is_boundary=s.is_boundary,
            )
            for s in z_supports
        ]

        self.num_x_stab = len(self.x_stabilizers)
        self.num_z_stab = len(self.z_stabilizers)

    def _generate_logical_operators(self) -> None:
        d = min(self.dx, self.dz)

        if self.rotated:
            logical_x_qubits = get_rotated_logical_x(d)
            logical_z_qubits = get_rotated_logical_z(d)
        else:
            logical_x_qubits = tuple(i * self.dz for i in range(self.dx))
            logical_z_qubits = tuple(range(self.dz))

        self.logical_x = LogicalOperator("X", logical_x_qubits)
        self.logical_z = LogicalOperator("Z", logical_z_qubits)

    @property
    def distance(self) -> int:
        """Code distance (minimum of dx and dz)."""
        return min(self.dx, self.dz)

    @property
    def num_ancilla(self) -> int:
        """Number of ancilla qubits (one per stabilizer)."""
        return self.num_x_stab + self.num_z_stab

    @property
    def num_qubits(self) -> int:
        """Total number of qubits (data + ancilla)."""
        return self.num_data + self.num_ancilla


class SurfacePatch:
    """A configurable surface code patch.

    Supports both rotated (default) and standard (non-rotated) layouts.

    Example:
        >>> patch = SurfacePatch.create(distance=5)  # Rotated (default)
        >>> patch = SurfacePatch.create(distance=5, rotated=False)  # Standard
        >>> patch = SurfacePatch.create(dx=3, dz=5)  # Asymmetric
    """

    def __init__(self, geometry: PatchGeometry) -> None:
        """Initialize a surface patch with the given geometry."""
        self.geometry = geometry

    @classmethod
    def create(
        cls,
        distance: int | None = None,
        dx: int | None = None,
        dz: int | None = None,
        orientation: PatchOrientation = PatchOrientation.X_TOP_BOTTOM,
        *,
        rotated: bool = True,
    ) -> SurfacePatch:
        """Create a surface code patch.

        Args:
            distance: Symmetric code distance (must be odd >= 3).
            dx: X distance for asymmetric codes.
            dz: Z distance for asymmetric codes.
            orientation: Patch boundary orientation.
            rotated: If True (default), use the rotated layout which is more
                common and uses fewer qubits. If False, use the standard
                (non-rotated) layout.
        """
        if distance is not None:
            if distance < 3 or distance % 2 == 0:
                msg = f"Distance must be odd >= 3, got {distance}"
                raise ValueError(msg)
            dx = dx or distance
            dz = dz or distance
        elif dx is not None and dz is not None:
            if dx < 3 or dx % 2 == 0:
                msg = f"dx must be odd >= 3, got {dx}"
                raise ValueError(msg)
            if dz < 3 or dz % 2 == 0:
                msg = f"dz must be odd >= 3, got {dz}"
                raise ValueError(msg)
        else:
            msg = "Must provide either distance or both dx and dz"
            raise ValueError(msg)

        geometry = PatchGeometry(dx=dx, dz=dz, orientation=orientation, rotated=rotated)
        return cls(geometry)

    @property
    def distance(self) -> int:
        """Code distance (minimum of dx and dz)."""
        return self.geometry.distance

    @property
    def dx(self) -> int:
        """X distance of the patch."""
        return self.geometry.dx

    @property
    def dz(self) -> int:
        """Z distance of the patch."""
        return self.geometry.dz

    @property
    def num_data(self) -> int:
        """Number of data qubits."""
        return self.geometry.num_data

    @property
    def num_qubits(self) -> int:
        """Total number of qubits (data + ancilla)."""
        return self.geometry.num_qubits

    @property
    def x_stabilizers(self) -> list[Stabilizer]:
        """X stabilizers of the patch."""
        return self.geometry.x_stabilizers

    @property
    def z_stabilizers(self) -> list[Stabilizer]:
        """Z stabilizers of the patch."""
        return self.geometry.z_stabilizers

    @property
    def rotated(self) -> bool:
        """True if using rotated layout, False for standard layout."""
        return self.geometry.rotated

    @property
    def num_ancilla(self) -> int:
        """Number of ancilla qubits."""
        return self.geometry.num_ancilla

    def get_patch_descriptor(self) -> SurfacePatchDescriptor:
        """Return a public metadata summary for this patch."""
        return {
            "distance": self.distance,
            "dx": self.dx,
            "dz": self.dz,
            "rotated": self.rotated,
            "orientation": self.geometry.orientation.name,
            "num_data": self.num_data,
            "num_ancilla": self.num_ancilla,
            "num_qubits": self.num_qubits,
        }

    def get_stabilizer_descriptor(
        self,
        stab_type: str,
        index: int,
    ) -> StabilizerDescriptor:
        """Return one public stabilizer descriptor."""
        stabs = self.x_stabilizers if stab_type.upper() == "X" else self.z_stabilizers
        stab = stabs[index]
        metadata = _get_stabilizer_schedule_metadata(stab, self)
        positions = [list(self.geometry.id_to_pos[q]) for q in stab.data_qubits]
        return {
            **metadata,
            "data_qubits": list(stab.data_qubits),
            "data_qubit_positions": positions,
            "weight": stab.weight,
        }

    def iter_stabilizer_descriptors(
        self,
        stab_type: str | None = None,
    ) -> list[StabilizerDescriptor]:
        """Iterate over public stabilizer descriptors."""
        if stab_type is None:
            descriptors: list[StabilizerDescriptor] = []
            descriptors.extend(self.iter_stabilizer_descriptors("X"))
            descriptors.extend(self.iter_stabilizer_descriptors("Z"))
            return descriptors

        kind = stab_type.upper()
        stabs = self.x_stabilizers if kind == "X" else self.z_stabilizers
        return [self.get_stabilizer_descriptor(kind, stab.index) for stab in stabs]

    def get_logical_descriptor(self, logical_type: str) -> LogicalDescriptor:
        """Return one public logical-operator descriptor."""
        kind = logical_type.upper()
        logical = self.geometry.logical_x if kind == "X" else self.geometry.logical_z
        if logical is None:
            msg = f"Logical operator {kind} is not available"
            raise ValueError(msg)

        positions = [list(self.geometry.id_to_pos[q]) for q in logical.data_qubits]
        rows = {row for row, _ in map(tuple, positions)}
        cols = {col for _, col in map(tuple, positions)}
        support_axis = "vertical" if len(cols) == 1 else "horizontal"
        if len(rows) == 1 and len(cols) != 1:
            support_axis = "horizontal"

        return {
            "logical_type": logical.op_type,
            "data_qubits": list(logical.data_qubits),
            "data_qubit_positions": positions,
            "weight": len(logical.data_qubits),
            "support_axis": support_axis,
        }

    def iter_logical_descriptors(self) -> list[LogicalDescriptor]:
        """Iterate over logical descriptors in X, Z order."""
        return [
            self.get_logical_descriptor("X"),
            self.get_logical_descriptor("Z"),
        ]

    def get_parity_matrix(self, stab_type: str) -> pecos.Array:
        """Get parity check matrix."""
        stabs = self.x_stabilizers if stab_type == "X" else self.z_stabilizers
        num_stab = len(stabs)
        matrix = zeros((num_stab, self.num_data), dtype="int64")

        for stab in stabs:
            for q in stab.data_qubits:
                matrix[stab.index, q] = 1

        return matrix


class SurfacePatchBuilder:
    """Builder for creating SurfacePatch instances.

    By default, creates a rotated surface code (more common). Use `.standard()`
    to create a non-rotated surface code.

    Example:
        >>> patch = SurfacePatchBuilder().with_distance(5).with_orientation(PatchOrientation.Z_TOP_BOTTOM).build()

        >>> # Non-rotated (standard) surface code:
        >>> patch = SurfacePatchBuilder().with_distance(5).standard().build()
    """

    def __init__(self) -> None:
        """Initialize the builder with default settings."""
        self._distance: int | None = None
        self._dx: int | None = None
        self._dz: int | None = None
        self._orientation: PatchOrientation = PatchOrientation.X_TOP_BOTTOM
        self._rotated: bool = True

    def with_distance(self, distance: int) -> SurfacePatchBuilder:
        """Set symmetric distance."""
        self._distance = distance
        return self

    def with_distances(self, dx: int, dz: int) -> SurfacePatchBuilder:
        """Set asymmetric distances."""
        self._dx = dx
        self._dz = dz
        return self

    def with_orientation(self, orientation: PatchOrientation) -> SurfacePatchBuilder:
        """Set patch orientation."""
        self._orientation = orientation
        return self

    def rotated(self) -> SurfacePatchBuilder:
        """Use rotated surface code layout (default).

        The rotated layout is more common and uses fewer physical qubits
        for the same code distance.
        """
        self._rotated = True
        return self

    def standard(self) -> SurfacePatchBuilder:
        """Use standard (non-rotated) surface code layout.

        The standard layout uses more physical qubits but may be preferred
        for certain applications or compatibility with existing code.
        """
        self._rotated = False
        return self

    def build(self) -> SurfacePatch:
        """Build the SurfacePatch."""
        return SurfacePatch.create(
            distance=self._distance,
            dx=self._dx,
            dz=self._dz,
            orientation=self._orientation,
            rotated=self._rotated,
        )
