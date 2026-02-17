# Copyright 2024 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Surface code patch with runtime configuration.

Provides a flexible, runtime-configurable surface code patch
with geometry stored as data structures.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

from pecos_rslib.num import zeros

from pecos.qec.surface.layouts import (
    compute_rotated_x_stabilizers,
    compute_rotated_z_stabilizers,
    compute_x_stabilizer_supports,
    compute_z_stabilizer_supports,
    get_rotated_logical_x,
    get_rotated_logical_z,
)

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
        """Number of ancilla qubits."""
        return 2

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
    ) -> "SurfacePatch":
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

    def get_parity_matrix(self, stab_type: str) -> "pecos.Array":
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

    def with_distance(self, distance: int) -> "SurfacePatchBuilder":
        """Set symmetric distance."""
        self._distance = distance
        return self

    def with_distances(self, dx: int, dz: int) -> "SurfacePatchBuilder":
        """Set asymmetric distances."""
        self._dx = dx
        self._dz = dz
        return self

    def with_orientation(self, orientation: PatchOrientation) -> "SurfacePatchBuilder":
        """Set patch orientation."""
        self._orientation = orientation
        return self

    def rotated(self) -> "SurfacePatchBuilder":
        """Use rotated surface code layout (default).

        The rotated layout is more common and uses fewer physical qubits
        for the same code distance.
        """
        self._rotated = True
        return self

    def standard(self) -> "SurfacePatchBuilder":
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
