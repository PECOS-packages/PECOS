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

"""Square lattice layout generation for surface codes.

This module re-exports the layout function from pecos.qec.surface.layouts.
"""

from pecos.qec.surface.layouts import generate_nonrotated_surface_layout


def gen_layout(
    width: int,
    height: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[list[tuple[int, int]]]]:
    """Generate rectangular surface code patch layout for a 4.4.4.4 lattice.

    This generates the non-rotated surface code layout.

    Args:
        width: Width of the patch in logical qubits.
        height: Height of the patch in logical qubits.

    Returns:
        A tuple containing:
        - nodes: List of (x, y) coordinates for data qubits.
        - dual_nodes: List of (x, y) coordinates for ancilla qubits.
        - polygons: List of polygons representing stabilizer checks.
    """
    return generate_nonrotated_surface_layout(width, height)
