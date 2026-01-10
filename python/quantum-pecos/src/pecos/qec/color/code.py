# Copyright 2024 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""4.8.8 Triangular Color Code implementation."""

from dataclasses import dataclass, field

import pecos
from pecos.qec.color.geometry import generate_488_layout
from pecos.qec.generic import StabilizerCheck


@dataclass(frozen=True)
class ColorCodeStabilizer:
    """A stabilizer in the color code.

    Attributes:
        index: Unique identifier
        qubits: Tuple of qubit indices
        color: 'red', 'green', or 'blue'
        is_boundary: Whether this is a boundary stabilizer
    """

    index: int
    qubits: tuple[int, ...]
    color: str
    is_boundary: bool = False

    @property
    def weight(self) -> int:
        """Number of qubits this stabilizer acts on."""
        return len(self.qubits)


@dataclass
class ColorCode488Geometry:
    """Geometry for a 4.8.8 color code.

    Stores the layout and stabilizer structure.
    """

    distance: int
    qubit_positions: dict[int, tuple[int, int]] = field(default_factory=dict)
    stabilizers: list[ColorCodeStabilizer] = field(default_factory=list)
    num_data: int = field(init=False)
    num_stabilizers: int = field(init=False)

    def __post_init__(self) -> None:
        """Generate the layout."""
        if self.distance < 3 or self.distance % 2 == 0:
            msg = f"Distance must be odd >= 3, got {self.distance}"
            raise ValueError(msg)

        nodeid2pos, polygons = generate_488_layout(self.distance)
        self.qubit_positions = nodeid2pos
        self.num_data = len(nodeid2pos)

        # Convert polygons to stabilizers
        self.stabilizers = []
        for i, polygon in enumerate(polygons):
            qubits = tuple(polygon[:-1])
            color = polygon[-1]
            is_boundary = len(qubits) < 4  # Boundary stabilizers have fewer qubits

            self.stabilizers.append(
                ColorCodeStabilizer(
                    index=i,
                    qubits=qubits,
                    color=color,
                    is_boundary=is_boundary,
                ),
            )

        self.num_stabilizers = len(self.stabilizers)


class ColorCode488:
    """A 4.8.8 triangular color code.

    The color code is a topological QEC code with three-colored
    stabilizers. It supports transversal Clifford gates.

    Example:
        >>> code = ColorCode488.create(distance=3)
        >>> print(f"Data qubits: {code.num_data}")
        >>> print(f"Stabilizers: {code.num_stabilizers}")
    """

    def __init__(self, geometry: ColorCode488Geometry) -> None:
        """Initialize with geometry."""
        self.geometry = geometry
        self._cache: dict[str, object] = {}

    @classmethod
    def create(cls, distance: int) -> "ColorCode488":
        """Create a color code with the given distance.

        Args:
            distance: Code distance (must be odd >= 3)

        Returns:
            ColorCode488 instance
        """
        geometry = ColorCode488Geometry(distance=distance)
        return cls(geometry)

    @property
    def distance(self) -> int:
        """Code distance."""
        return self.geometry.distance

    @property
    def num_data(self) -> int:
        """Number of data qubits."""
        return self.geometry.num_data

    @property
    def num_stabilizers(self) -> int:
        """Number of stabilizers."""
        return self.geometry.num_stabilizers

    @property
    def qubit_positions(self) -> dict[int, tuple[int, int]]:
        """Qubit position mapping."""
        return self.geometry.qubit_positions

    @property
    def stabilizers(self) -> list[ColorCodeStabilizer]:
        """List of stabilizers."""
        return self.geometry.stabilizers

    def get_stabilizers_by_color(self, color: str) -> list[ColorCodeStabilizer]:
        """Get stabilizers of a specific color.

        Args:
            color: 'red', 'green', or 'blue'

        Returns:
            List of stabilizers with that color
        """
        return [s for s in self.stabilizers if s.color == color]

    def get_red_stabilizers(self) -> list[ColorCodeStabilizer]:
        """Get red (weight-4) stabilizers."""
        return self.get_stabilizers_by_color("red")

    def get_green_stabilizers(self) -> list[ColorCodeStabilizer]:
        """Get green stabilizers."""
        return self.get_stabilizers_by_color("green")

    def get_blue_stabilizers(self) -> list[ColorCodeStabilizer]:
        """Get blue stabilizers."""
        return self.get_stabilizers_by_color("blue")

    def get_parity_matrix(self) -> pecos.Array:
        """Get the parity check matrix.

        Returns:
            PECOS array of shape (num_stabilizers, num_data)
        """
        matrix = pecos.zeros((self.num_stabilizers, self.num_data), dtype="int64")

        for stab in self.stabilizers:
            for q in stab.qubits:
                matrix[stab.index, q] = 1

        return matrix

    def get_logical_x(self) -> tuple[int, ...]:
        """Get logical X operator qubits.

        For the triangular color code, logical X runs along one boundary.
        """
        # Bottom row of qubits forms logical X
        positions = self.qubit_positions

        # Find qubits with lowest y coordinate
        min_y = min(pos[1] for pos in positions.values())
        logical_qubits = [qid for qid, pos in positions.items() if pos[1] == min_y]

        return tuple(sorted(logical_qubits))

    def get_logical_z(self) -> tuple[int, ...]:
        """Get logical Z operator qubits.

        For the triangular 4.8.8 color code, both logical X and logical Z
        are supported on the same qubits (the bottom boundary), but with
        different Pauli types (X vs Z). This is because the color code
        is self-dual.
        """
        # Same support as logical X for self-dual color code
        return self.get_logical_x()

    def to_generic_checks(self) -> list[StabilizerCheck]:
        """Convert to generic StabilizerCheck objects.

        Returns:
            List of StabilizerCheck instances for use with generic framework
        """
        checks = []
        for stab in self.stabilizers:
            check = StabilizerCheck.x_check(
                index=stab.index,
                qubits=stab.qubits,
                is_boundary=stab.is_boundary,
            )
            # Note: Color codes measure both X and Z on same qubits
            # This returns X checks; for Z checks, create z_check versions
            checks.append(check)
        return checks


class ColorCode488Builder:
    """Builder for creating ColorCode488 instances.

    Example:
        >>> code = ColorCode488Builder().with_distance(5).build()
    """

    def __init__(self) -> None:
        """Initialize the builder."""
        self._distance: int | None = None

    def with_distance(self, distance: int) -> "ColorCode488Builder":
        """Set the code distance."""
        self._distance = distance
        return self

    def build(self) -> ColorCode488:
        """Build the ColorCode488."""
        if self._distance is None:
            msg = "Distance must be set before building"
            raise ValueError(msg)
        return ColorCode488.create(distance=self._distance)
