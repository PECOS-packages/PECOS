"""Visualization strategy for rotated lattice surface codes."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos.slr.qeclib.surface.patches.patch_base import SurfacePatch
    from pecos.slr.qeclib.surface.visualization.visualization_base import VisData


class RotatedLatticeVisualization:
    """Visualization for rotated square lattice surface codes."""

    @staticmethod
    def get_visualization_data(patch: "SurfacePatch") -> "VisData":
        """Get visualization data for the patch."""
        return patch.layout.get_visualization_elements(
            patch.dx,
            patch.dz,
            patch.stab_gens,
        )

    @staticmethod
    def supports_view(view_type: str) -> bool:
        """Check if the visualization supports the given view type."""
        return view_type == "lattice"
