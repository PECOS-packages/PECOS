"""Base classes for surface code layouts."""

from enum import Enum
from typing import Protocol

from pecos.slr.qeclib.surface.visualization.visualization_base import VisualizationData


class LatticeType(Enum):
    """Lattices that the patches of the surface code can be constructed from.

    References:
        1. Jonas Anderson, "Fault-tolerance in two-dimensional topological systems" by
           <https://digitalrepository.unm.edu/phyc_etds/4/>
    """

    SQUARE = (4, 4, 4, 4)
    RHOMBITRIHEXAGONAL = (3, 4, 6, 4)
    TRIHEXAGONAL = (3, 6, 3, 6)


class Layout(Protocol):
    """Protocol for different layout strategies."""

    def get_stabilizers_gens(
        self,
        dx: int,
        dz: int,
    ) -> list[tuple[str, tuple[int, ...]]]:
        """Get stabilizer generators for the layout."""
        ...

    def get_data_positions(self, dx: int, dz: int) -> list[tuple[int, int]]:
        """Get positions of data qubits in the layout."""
        ...

    def validate_dimensions(self, dx: int, dz: int) -> None:
        """Validate the layout dimensions."""
        ...

    def get_visualization_elements(
        self,
        dx: int,
        dz: int,
        stab_gens: list[tuple[str, tuple[int, ...]]],
    ) -> VisualizationData:
        """Get visualization elements for the layout."""
        ...
