# Copyright 2024 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Color 488 quantum error correction code implementation."""

from typing import Any

from pecos.slr.qeclib.color488.abstract_layout import gen_layout, get_boundaries


class Color488:
    """Implementation of the Color 488 quantum error correction code."""

    def __init__(self, distance: int) -> None:
        """Initialize a Color 488 code instance.

        Args:
            distance: The code distance.
        """
        self.distance = distance
        self._layout_cache: (
            tuple[dict[int, tuple[int, int]], list[list[Any]]] | None
        ) = None

    def get_layout(self) -> tuple[dict[int, tuple[int, int]], list[list[Any]]]:
        """Get the layout of the color 488 code.

        Returns:
            A tuple containing:
            - nodeid2pos: Mapping from node IDs to positions.
            - polygons: List of stabilizer polygons.
        """
        if self._layout_cache is None:
            self._layout_cache = gen_layout(self.distance)
        return self._layout_cache

    def get_boundaries(self) -> tuple[list[int], list[int], list[int]]:
        """Get the boundaries of the color 488 code layout.

        Returns:
            A tuple containing the left, bottom, and right boundary node lists.
        """
        nodeid2pos, _ = self.get_layout()
        return get_boundaries(nodeid2pos)

    def num_data_qubits(self) -> int:
        """Get the number of data qubits in the color 488 code.

        Returns:
            int: The number of data qubits.
        """
        nodes, _ = self.get_layout()
        return len(nodes)
