# Copyright 2018 The PECOS Developers
# Copyright 2018 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract
# DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Rotated square lattice layout for surface codes.

This module provides implementations for creating and visualizing rotated square lattice
layouts (4.4.4.4 lattice) used in quantum error correction surface codes.
"""


from pecos.qeclib.surface.visualization.visualization_base import VisualizationData


class SquareRotatedLayout:
    """4.4.4.4 rotated lattice implementation."""

    @staticmethod
    def get_stabilizers_gens(dx: int, dz: int) -> list[tuple[str, tuple[int, ...]]]:
        """Get stabilizer generators for the rotated square lattice.

        Args:
            dx: The x-dimension of the lattice.
            dz: The z-dimension of the lattice.

        Returns:
            List of tuples containing Pauli type ('X' or 'Z') and qubit indices.
        """
        return get_stab_gens(dx, dz)

    @staticmethod
    def get_data_positions(dx: int, dz: int) -> list[tuple[int, int]]:
        """Get positions of data qubits in the lattice.

        Args:
            dx: The x-dimension of the lattice.
            dz: The z-dimension of the lattice.

        Returns:
            List of (x, y) coordinate tuples for each data qubit.
        """
        return [calc_id2pos(i, dz, dx) for i in range(dx * dz)]

    @staticmethod
    def validate_dimensions(dx: int, dz: int) -> None:
        """Validate lattice dimensions.

        Args:
            dx: The x-dimension of the lattice.
            dz: The z-dimension of the lattice.

        Raises:
            ValueError: If either dimension is less than 1.
        """
        if dx < 1 or dz < 1:
            msg = "Dimensions must be at least 1"
            raise ValueError(msg)

    @staticmethod
    def get_visualization_elements(
        dx: int,
        dz: int,
        stab_gens: list[tuple[str, tuple[int, ...]]],
    ) -> VisualizationData:
        """Get visualization elements for rendering the lattice.

        Args:
            dx: The x-dimension of the lattice.
            dz: The z-dimension of the lattice.
            stab_gens: List of stabilizer generators.

        Returns:
            VisualizationData containing nodes, polygons, and color information.
        """
        polygon_colors = {}
        for i, (pauli, _) in enumerate(stab_gens):
            polygon_colors[i] = 0 if pauli == "X" else 1

        polygons = [
            [calc_id2pos(id_, dz, dx) for id_ in datas] for _, datas in stab_gens
        ]

        polygons = [order_coords_counter_clockwise(coords) for coords in polygons]

        for coords in polygons:
            # make a triangle to form diagons
            if len(coords) == 2:
                # Work out the original (x, y) of the dual node
                (x1, y1), (x2, y2) = coords
                if y1 == y2 == 1:
                    coords.insert(0, (x1 + 1, 0))
                elif y1 == y2 == 2 * dx - 1:
                    coords.insert(0, (x1 + 1, y1 + 1))
                elif x1 == x2 == 1:
                    coords.insert(0, (x1 - 1, y1 - 1))
                elif x1 == x2 == 2 * dz - 1:
                    coords.insert(0, (x1 + 1, y1 + 1))
                else:
                    msg = f"Unexpected digon coordinates: {coords}"
                    raise Exception(msg)

        nodes = [calc_id2pos(i, dz, dx) for i in range(dx * dz)]

        return VisualizationData(
            nodes=nodes,
            polygons=polygons,
            polygon_colors=polygon_colors,
        )


def calc_id2pos(i: int, width: int, height: int) -> tuple[int, int]:
    """Convert qubit ID to position coordinates.

    Args:
        i: The qubit ID.
        width: The width of the lattice.
        height: The height of the lattice.

    Returns:
        Tuple of (x, y) coordinates.
    """
    # return (1+i*2)%(dz*2), (dx-(i//dz))*2-1
    return (1 + i * 2) % (width * 2), (height - (i // width)) * 2 - 1


def calc_pos2id(x: int, y: int, width: int, height: int) -> int:
    """Convert position coordinates to qubit ID.

    Args:
        x: The x-coordinate.
        y: The y-coordinate.
        width: The width of the lattice.
        height: The height of the lattice.

    Returns:
        The qubit ID.
    """
    # return (x-1)//2+((2*dx-y-1)//2)*dz
    return (x - 1) // 2 + ((2 * height - y - 1) // 2) * width


def get_stab_gens(height: int, width: int) -> list[tuple[str, tuple[int, ...]]]:
    """Generate rectangular rotated surface code patch layout for a 4.4.4.4 lattice."""
    lattice_height = height * 2
    lattice_width = width * 2

    polygons_0 = []
    polygons_1 = []

    for x in range(lattice_width + 1):
        for y in range(lattice_height + 1):
            if 0 < x < lattice_width and 0 < y < lattice_height:
                # Interior

                if x % 2 == 1 and y % 2 == 1:  # That is, both coordinates are odd...
                    pass

                elif x % 2 == 0 and y % 2 == 0:
                    # Bulk checks
                    poly = [
                        calc_pos2id(x - 1, y + 1, width, height),
                        calc_pos2id(x + 1, y + 1, width, height),
                        calc_pos2id(x - 1, y - 1, width, height),
                        calc_pos2id(x + 1, y - 1, width, height),
                    ]

                    if ((x + y) / 2) % 2 == 0:
                        polygons_0.append(poly)
                    else:
                        polygons_1.append(poly)

            elif 0 < x < lattice_width or 0 < y < lattice_height:
                # Not the corners or the interior

                if y == 0:
                    # Bottom: X checks

                    if x != 0 and x % 4 == 0:
                        poly = [
                            calc_pos2id(x - 1, y + 1, width, height),
                            calc_pos2id(x + 1, y + 1, width, height),
                        ]
                        polygons_0.append(poly)

                elif x == 0 and (y - 2) % 4 == 0:
                    # Left: Z checks
                    poly = [
                        calc_pos2id(x + 1, y + 1, width, height),
                        calc_pos2id(x + 1, y - 1, width, height),
                    ]
                    polygons_1.append(poly)

                if y == lattice_height:
                    # Top: X checks

                    if height % 2 == 0:
                        if x != 0 and x % 4 == 0:
                            poly = [
                                calc_pos2id(x - 1, y - 1, width, height),
                                calc_pos2id(x + 1, y - 1, width, height),
                            ]
                            polygons_0.append(poly)

                    elif (x - 2) % 4 == 0:
                        poly = [
                            calc_pos2id(x - 1, y - 1, width, height),
                            calc_pos2id(x + 1, y - 1, width, height),
                        ]
                        polygons_0.append(poly)

                elif x == lattice_width:
                    # Right: Z checks

                    if width % 2 == 1:
                        if y != 0 and y % 4 == 0:
                            poly = [
                                calc_pos2id(x - 1, y + 1, width, height),
                                calc_pos2id(x - 1, y - 1, width, height),
                            ]
                            polygons_1.append(poly)
                    elif (y - 2) % 4 == 0:
                        poly = [
                            calc_pos2id(x - 1, y + 1, width, height),
                            calc_pos2id(x - 1, y - 1, width, height),
                        ]
                        polygons_1.append(poly)

    return [("X", tuple(poly)) for poly in polygons_0] + [
        ("Z", tuple(poly)) for poly in polygons_1
    ]


def order_coords_counter_clockwise(
    coords: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Reorders a list of coordinates in approximate counter-clockwise order using x, y sorting.

    Parameters:
        coords (list): List of (x, y) tuples.

    Returns:
        list: List of (x, y) tuples ordered counter-clockwise.
    """
    if len(coords) < 3:
        return coords  # No reordering needed for lines or single points

    # Calculate centroid
    cx = sum(x for x, y in coords) / len(coords)
    cy = sum(y for x, y in coords) / len(coords)

    # Sort based on quadrant and relative position
    def sort_key(point: tuple[int, int]) -> tuple[int, int | float]:
        x, y = point
        if x >= cx and y >= cy:  # Top-right
            return 0, x
        if x < cx and y >= cy:  # Top-left
            return 1, -y
        if x < cx and y < cy:  # Bottom-left
            return 2, -x
        # Bottom-right
        return 3, y

    return sorted(coords, key=sort_key)
