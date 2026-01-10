"""Concrete implementations of surface code patches."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pecos.slr.qeclib.surface.layouts.rot_square_lattice import SquareRotatedLayout
from pecos.slr.qeclib.surface.patches.patch_base import BaseSurfacePatch

if TYPE_CHECKING:
    from pecos.slr.qeclib.surface.layouts.layout_base import Layout
    from pecos.slr.qeclib.surface.patches.patch_base import SurfacePatchOrientation


class RotatedSurfacePatch(BaseSurfacePatch):
    """Rotated surface patch."""

    def __init__(
        self,
        dx: int,
        dz: int,
        orientation: SurfacePatchOrientation,
        layout: Layout | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize a rotated surface code patch.

        Args:
            dx: Distance of the X logical operator.
            dz: Distance of the Z logical operator.
            orientation: Patch orientation determining boundary conditions.
            layout: Optional custom layout. Uses SquareRotatedLayout by default.
            name: Optional custom name for the patch.
        """
        super().__init__(dx, dz, orientation, name)

        # TODO: Should each surface patch carry this or should it be stored somewhere for reuse...
        #       or cached somehow
        if layout is None:
            layout = SquareRotatedLayout()
        self.layout = layout
        self.stab_gens = self.layout.get_stabilizers_gens(self.dx, self.dz)

    def _calculate_qubit_count(self) -> int:
        return self.dx * self.dz


class NonRotatedSurfacePatch(BaseSurfacePatch):
    """Standard surface patch."""

    def _calculate_qubit_count(self) -> int:
        # TODO: fix for non-rotated surface code
        return self.dx * self.dz
