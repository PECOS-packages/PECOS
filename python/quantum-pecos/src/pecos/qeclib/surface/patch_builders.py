"""Builder pattern implementation for surface code patch configuration."""

from typing import TYPE_CHECKING, TypeVar

from pecos.exceptions import ConfigurationError
from pecos.qeclib.surface.layouts.layout_base import LatticeType
from pecos.qeclib.surface.patches.patch_base import SurfacePatchOrientation
from pecos.qeclib.surface.patches.surface_patches import (
    NonRotatedSurfacePatch,
    RotatedSurfacePatch,
)

if TYPE_CHECKING:
    from pecos.qeclib.surface.patches.patch_base import SurfacePatch

Self = TypeVar("Self")


class SurfacePatchBuilder:
    """Build for complex patch configurations."""

    def __init__(self) -> None:
        """Initialize a new surface patch builder with default settings."""
        self.name: str | None = None
        self.dx: int | None = None
        self.dz: int | None = None
        self.is_rotated: bool = True
        self.lattice_type = LatticeType.SQUARE
        self.orientation = SurfacePatchOrientation.X_TOP_BOTTOM

    def set_name(self, name: str) -> Self:
        """Set a custom name for the patch."""
        self.name = name
        return self

    def with_distances(self, dx: int, dz: int) -> Self:
        """Set the X and Z code distances.

        Args:
            dx: X distance for detecting/correcting Z errors
            dz: Z distance for detecting/correcting X errors

        The overall code distance is min(dx, dz).
        """
        if dx < 1 or dz < 1:
            msg = f"Distances must be positive, got dx={dx}, dz={dz}"
            raise ConfigurationError(msg)
        self.dx = dx
        self.dz = dz
        return self

    def not_rotated(self) -> Self:
        """Configure as non-rotated surface code."""
        self.is_rotated = False
        return self

    def with_orientation(
        self,
        orientation: SurfacePatchOrientation,
    ) -> Self:
        """Set the patch orientation."""
        self.orientation = orientation
        return self

    def with_lattice(self, lattice: LatticeType) -> Self:
        """Set the lattice type for the surface code patch.

        Args:
            lattice: The type of lattice to use (e.g., SQUARE).

        Returns:
            Self: The builder instance for method chaining.
        """
        self.lattice_type = lattice
        return self

    def build(self) -> "SurfacePatch":
        """Create the surface code patch with the configured settings."""
        # Validate configuration
        if self.dx is None or self.dz is None:
            msg = "Must specify distance(s)"
            raise ConfigurationError(msg)

        if self.lattice_type != LatticeType.SQUARE:
            msg = "Currently only Lattice type SQUARE is supported"
            raise NotImplementedError(msg)

        if self.dx < 1 or self.dz < 1:
            msg = "The x and z distances must be at least 1."
            raise ConfigurationError(msg)

        if self.is_rotated:
            return RotatedSurfacePatch(
                name=self.name,
                dx=self.dx,
                dz=self.dz,
                orientation=self.orientation,
            )
        return NonRotatedSurfacePatch(
            name=self.name,
            dx=self.dx,
            dz=self.dz,
            orientation=self.orientation,
        )
