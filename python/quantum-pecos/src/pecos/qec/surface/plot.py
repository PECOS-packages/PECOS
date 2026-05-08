# Copyright 2024 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Visualization for surface code patches.

Bridges `pecos.qec.surface.SurfacePatch` geometry into the
`plot_colored_polygons` renderer from the SLR visualization module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pecos.qec.surface.layouts.rotated_lattice import rotated_id_to_position
from pecos.qec.surface.schedule import get_stab_schedule
from pecos.slr.qeclib.surface.visualization.lattice_2d import (
    Lattice2DConfig,
    plot_colored_polygons,
)

if TYPE_CHECKING:
    from matplotlib import pyplot as plt

    from pecos.qec.surface.patch import SurfacePatch


def _order_counter_clockwise(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Reorder coordinates in counter-clockwise order around their centroid."""
    if len(coords) < 3:
        return coords

    cx = sum(x for x, y in coords) / len(coords)
    cy = sum(y for x, y in coords) / len(coords)

    def sort_key(point: tuple[int, int]) -> tuple[int, int | float]:
        x, y = point
        if x >= cx and y >= cy:
            return 0, x
        if x < cx and y >= cy:
            return 1, -y
        if x < cx and y < cy:
            return 2, -x
        return 3, y

    return sorted(coords, key=sort_key)


def _make_triangle(coords: list[tuple[int, int]], d: int) -> None:
    """Convert a 2-qubit boundary digon into a triangle by inserting a third vertex.

    Follows the convention from rot_square_lattice.py: identify which
    boundary the digon sits on and add a point outside the data qubit grid.
    """
    if len(coords) != 2:
        return

    (x1, y1), (x2, y2) = coords
    if y1 == y2 == 1:
        # Bottom boundary
        coords.insert(0, (x1 + 1, 0))
    elif y1 == y2 == 2 * d - 1:
        # Top boundary
        coords.insert(0, (x1 + 1, y1 + 1))
    elif x1 == x2 == 1:
        # Left boundary
        coords.insert(0, (x1 - 1, y1 - 1))
    elif x1 == x2 == 2 * d - 1:
        # Right boundary
        coords.insert(0, (x1 + 1, y1 + 1))
    else:
        msg = f"Unexpected digon coordinates: {coords}"
        raise ValueError(msg)


def plot_patch(
    patch: SurfacePatch,
    *,
    show_cnot_order: bool = False,
    config: Lattice2DConfig | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a surface code patch showing stabilizer plaquettes and data qubits.

    X stabilizers are shown in red, Z stabilizers in blue.
    Data qubits are numbered by their qubit ID.

    Args:
        patch: A SurfacePatch instance.
        show_cnot_order: If True, annotate each plaquette with numbered
            labels showing the CNOT gate ordering (the order of data_qubits
            in each stabilizer).
        config: Optional Lattice2DConfig for styling. Uses defaults if None.

    Returns:
        Tuple of (Figure, Axes) from matplotlib.
    """
    d = patch.distance

    # Build node list: data qubit positions in row-major order
    nodes = [rotated_id_to_position(i, d) for i in range(patch.num_data)]

    # Build polygons and color map
    polygons: list[list[tuple[int, int]]] = []
    polygon_colors: dict[int, int] = {}

    all_stabilizers = list(patch.x_stabilizers) + list(patch.z_stabilizers)

    for poly_idx, stab in enumerate(all_stabilizers):
        coords = [rotated_id_to_position(q, d) for q in stab.data_qubits]
        coords = _order_counter_clockwise(coords)
        _make_triangle(coords, d)
        polygons.append(coords)
        # 0 = X (red), 1 = Z (blue)
        polygon_colors[poly_idx] = 0 if stab.stab_type == "X" else 1

    if config is None:
        config = Lattice2DConfig()

    fig, ax = plot_colored_polygons(
        polygons=polygons,
        points_to_plot=nodes,
        polygon_colors=polygon_colors,
        config=config,
    )

    if show_cnot_order:
        _annotate_cnot_order(ax, all_stabilizers, d)

    return fig, ax


def _annotate_cnot_order(ax: plt.Axes, stabilizers: list, d: int) -> None:
    """Add numbered labels showing CNOT round (1-4) for each data qubit in each stabilizer."""
    for stab in stabilizers:
        schedule = get_stab_schedule(
            stab.stab_type,
            stab.data_qubits,
            stab.is_boundary,
            d,
            d,
        )

        # Compute centroid of the stabilizer
        positions = [rotated_id_to_position(q, d) for q in stab.data_qubits]
        centroid_x = sum(x for x, y in positions) / len(positions)
        centroid_y = sum(y for x, y in positions) / len(positions)

        for rnd_0based, data_q in schedule:
            px, py = rotated_id_to_position(data_q, d)
            # Place label partway between the data qubit and the centroid
            lx = px + 0.4 * (centroid_x - px)
            ly = py + 0.4 * (centroid_y - py)
            ax.text(
                lx,
                ly,
                str(rnd_0based + 1),
                color="black",
                fontsize=7,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=5,
                bbox={
                    "boxstyle": "round,pad=0.1",
                    "facecolor": "white",
                    "alpha": 0.7,
                    "edgecolor": "none",
                },
            )


def plot_surface_code(
    d: int,
    *,
    show_cnot_order: bool = False,
    config: Lattice2DConfig | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Create and plot a rotated surface code patch of the given distance.

    Convenience function that creates a SurfacePatch and calls plot_patch.

    Args:
        d: Code distance (must be odd >= 3).
        show_cnot_order: If True, show CNOT ordering within stabilizers.
        config: Optional Lattice2DConfig for styling.

    Returns:
        Tuple of (Figure, Axes) from matplotlib.
    """
    from pecos.qec.surface.patch import SurfacePatch

    patch = SurfacePatch.create(distance=d)
    return plot_patch(patch, show_cnot_order=show_cnot_order, config=config)
