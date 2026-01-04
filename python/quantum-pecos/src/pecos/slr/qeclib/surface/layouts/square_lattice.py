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

"""Square lattice layout generation for surface codes."""


def gen_layout(
    width: int,
    height: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[list[tuple[int, int]]]]:
    """Generate rectangular surface code patch layout for a 4.4.4.4 lattice.

    Args:
        width: Width of the patch in logical qubits.
        height: Height of the patch in logical qubits.

    Returns:
        A tuple containing:
        - nodes: List of (x, y) coordinates for data qubits.
        - dual_nodes: List of (x, y) coordinates for ancilla qubits.
        - polygons: List of polygons representing stabilizer checks.
    """
    lattice_height = 2 * (height - 1)
    lattice_width = 2 * (width - 1)

    nodes = []
    dual_nodes = []
    polygons_0 = []
    polygons_1 = []

    # Determine the position of things
    for y in range(lattice_height + 1):
        for x in range(lattice_width + 1):
            if (x % 2 == 0 and y % 2 == 0) or (x % 2 == 1 and y % 2 == 1):
                # Data
                nodes.append((x, y))

            elif x % 2 == 1 and y % 2 == 0:
                # X ancilla
                dual_nodes.append((x, y))

                poly = []
                if y != lattice_height:
                    poly.append((x, y + 1))
                if x != 0:
                    poly.append((x - 1, y))
                if y != 0:
                    poly.append((x, y - 1))
                if x != lattice_width:
                    poly.append((x + 1, y))

                polygons_0.append(poly)

            elif x % 2 == 0 and y % 2 == 1:
                # Z ancilla
                dual_nodes.append((x, y))

                poly = []
                if y != lattice_height:
                    poly.append((x, y + 1))
                if x != 0:
                    poly.append((x - 1, y))
                if y != 0:
                    poly.append((x, y - 1))
                if x != lattice_width:
                    poly.append((x + 1, y))

                polygons_0.append(poly)

    polygons = []
    polygons.extend(polygons_0)
    polygons.extend(polygons_1)

    return nodes, dual_nodes, polygons
