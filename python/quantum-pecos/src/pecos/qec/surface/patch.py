# Copyright 2024 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Surface code patch with runtime configuration.

Provides a flexible, runtime-configurable surface code patch
with geometry stored as data structures.
"""

from dataclasses import dataclass, field
from enum import Enum, auto

from pecos.qec.surface.layouts import (
    compute_x_stabilizer_supports,
    compute_z_stabilizer_supports,
)


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
        return len(self.data_qubits)


@dataclass(frozen=True)
class LogicalOperator:
    """A logical operator for the surface code."""

    op_type: str
    data_qubits: tuple[int, ...]


@dataclass
class PatchGeometry:
    """Geometry of a surface code patch."""

    dx: int
    dz: int
    orientation: PatchOrientation = PatchOrientation.X_TOP_BOTTOM

    n_data: int = field(init=False)
    n_x_stab: int = field(init=False)
    n_z_stab: int = field(init=False)

    pos_to_id: dict[tuple[int, int], int] = field(default_factory=dict)
    id_to_pos: dict[int, tuple[int, int]] = field(default_factory=dict)

    x_stabilizers: list[Stabilizer] = field(default_factory=list)
    z_stabilizers: list[Stabilizer] = field(default_factory=list)

    logical_x: LogicalOperator | None = None
    logical_z: LogicalOperator | None = None

    def __post_init__(self):
        self.n_data = self.dx * self.dz
        self.n_x_stab = (self.dx * self.dz - 1) // 2
        self.n_z_stab = (self.dx * self.dz - 1) // 2

        self._generate_layout()
        self._generate_stabilizers()
        self._generate_logical_operators()

    def _generate_layout(self):
        for row in range(self.dx):
            for col in range(self.dz):
                idx = row * self.dz + col
                pos = (row, col)
                self.pos_to_id[pos] = idx
                self.id_to_pos[idx] = pos

    def _generate_stabilizers(self):
        d = min(self.dx, self.dz)

        x_supports = compute_x_stabilizer_supports(d)
        z_supports = compute_z_stabilizer_supports(d)

        self.x_stabilizers = [
            Stabilizer(
                index=s.index,
                stab_type='X',
                data_qubits=s.data_qubits,
                is_boundary=s.is_boundary,
            )
            for s in x_supports
        ]

        self.z_stabilizers = [
            Stabilizer(
                index=s.index,
                stab_type='Z',
                data_qubits=s.data_qubits,
                is_boundary=s.is_boundary,
            )
            for s in z_supports
        ]

        self.n_x_stab = len(self.x_stabilizers)
        self.n_z_stab = len(self.z_stabilizers)

    def _generate_logical_operators(self):
        logical_x_qubits = tuple(i * self.dz for i in range(self.dx))
        self.logical_x = LogicalOperator('X', logical_x_qubits)

        logical_z_qubits = tuple(range(self.dz))
        self.logical_z = LogicalOperator('Z', logical_z_qubits)

    @property
    def distance(self) -> int:
        return min(self.dx, self.dz)

    @property
    def n_ancilla(self) -> int:
        return 2

    @property
    def n_qubits(self) -> int:
        return self.n_data + self.n_ancilla


class SurfacePatch:
    """A configurable surface code patch.

    Example:
        >>> patch = SurfacePatch.create(distance=5)
        >>> patch = SurfacePatch.create(dx=3, dz=5)  # Asymmetric
    """

    def __init__(self, geometry: PatchGeometry):
        self.geometry = geometry

    @classmethod
    def create(
        cls,
        distance: int | None = None,
        dx: int | None = None,
        dz: int | None = None,
        orientation: PatchOrientation = PatchOrientation.X_TOP_BOTTOM,
    ) -> "SurfacePatch":
        """Create a surface code patch."""
        if distance is not None:
            if distance < 3 or distance % 2 == 0:
                raise ValueError(f"Distance must be odd >= 3, got {distance}")
            dx = dx or distance
            dz = dz or distance
        elif dx is not None and dz is not None:
            if dx < 3 or dx % 2 == 0:
                raise ValueError(f"dx must be odd >= 3, got {dx}")
            if dz < 3 or dz % 2 == 0:
                raise ValueError(f"dz must be odd >= 3, got {dz}")
        else:
            raise ValueError("Must provide either distance or both dx and dz")

        geometry = PatchGeometry(dx=dx, dz=dz, orientation=orientation)
        return cls(geometry)

    @property
    def distance(self) -> int:
        return self.geometry.distance

    @property
    def dx(self) -> int:
        return self.geometry.dx

    @property
    def dz(self) -> int:
        return self.geometry.dz

    @property
    def n_data(self) -> int:
        return self.geometry.n_data

    @property
    def n_qubits(self) -> int:
        return self.geometry.n_qubits

    @property
    def x_stabilizers(self) -> list[Stabilizer]:
        return self.geometry.x_stabilizers

    @property
    def z_stabilizers(self) -> list[Stabilizer]:
        return self.geometry.z_stabilizers

    def get_parity_matrix(self, stab_type: str):
        """Get parity check matrix."""
        import pecos

        stabs = self.x_stabilizers if stab_type == 'X' else self.z_stabilizers
        n_stab = len(stabs)
        matrix = pecos.zeros((n_stab, self.n_data), dtype="int64")

        for stab in stabs:
            for q in stab.data_qubits:
                matrix[stab.index, q] = 1

        return matrix


class SurfacePatchBuilder:
    """Builder for creating SurfacePatch instances.

    Example:
        >>> patch = (
        ...     SurfacePatchBuilder()
        ...     .with_distance(5)
        ...     .with_orientation(PatchOrientation.Z_TOP_BOTTOM)
        ...     .build()
        ... )
    """

    def __init__(self):
        self._distance: int | None = None
        self._dx: int | None = None
        self._dz: int | None = None
        self._orientation: PatchOrientation = PatchOrientation.X_TOP_BOTTOM

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

    def build(self) -> SurfacePatch:
        """Build the SurfacePatch."""
        return SurfacePatch.create(
            distance=self._distance,
            dx=self._dx,
            dz=self._dz,
            orientation=self._orientation,
        )
