"""Tests for surface patch construction and rendering."""

from pecos.qeclib.surface import (
    Lattice2DView,
    LatticeType,
    NonRotatedSurfacePatch,
    RotatedSurfacePatch,
    SurfacePatchBuilder,
    SurfacePatchOrientation,
)
from pecos.slr import Main, SlrConverter


def test_default_rot_surface_patch() -> None:
    """Test creating a default rotated surface patch."""
    prog = Main(
        s := RotatedSurfacePatch.default(3),
    )
    assert isinstance(s, RotatedSurfacePatch)
    SlrConverter(prog).qasm()


def test_default_rot_surface_patch_name() -> None:
    """Test creating a default rotated surface patch with custom name."""
    prog = Main(
        s := RotatedSurfacePatch.default(3, "s"),
    )
    assert isinstance(s, RotatedSurfacePatch)
    SlrConverter(prog).qasm()


def test_build_surface_patch() -> None:
    """Test building a non-rotated surface patch with custom parameters."""
    prog = Main(
        s := (
            SurfacePatchBuilder()
            .set_name("s")
            .with_distances(3, 5)
            .with_lattice(LatticeType.SQUARE)
            .with_orientation(SurfacePatchOrientation.Z_TOP_BOTTOM)
            .not_rotated()
            .build()
        ),
    )
    assert isinstance(s, NonRotatedSurfacePatch)
    SlrConverter(prog).qasm()


def test_build_rot_surface_patch() -> None:
    """Test building a rotated surface patch with custom parameters."""
    prog = Main(
        s := (
            SurfacePatchBuilder()
            .set_name("s")
            .with_distances(3, 5)
            .with_lattice(LatticeType.SQUARE)
            .with_orientation(SurfacePatchOrientation.Z_TOP_BOTTOM)
            .build()
        ),
    )
    assert isinstance(s, RotatedSurfacePatch)
    SlrConverter(prog).qasm()


def test_surface_patch_builder_render() -> None:
    """Test rendering a surface patch built with the builder."""
    s = SurfacePatchBuilder().with_distances(3, 3).build()
    Lattice2DView.render(s)


def test_rot_surface_patch_render() -> None:
    """Test rendering a default rotated surface patch."""
    s = RotatedSurfacePatch.default(3)
    Lattice2DView.render(s)
