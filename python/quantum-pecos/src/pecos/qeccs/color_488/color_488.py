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

"""Color 4.8.8 quantum error correcting code implementation.

This module provides the Color 4.8.8 topological quantum error correcting code,
which is based on a 4.8.8 lattice structure.
"""

from typing import Any

from pecos.qec.color.geometry import generate_488_layout
from pecos.qeccs.color_488.circuit_implementation1 import OneAncillaPerCheck
from pecos.qeccs.color_488.gates import GateIdentity, GateInitPlus, GateInitZero
from pecos.qeccs.color_488.instructions import (
    InstrInitPlus,
    InstrInitZero,
    InstrSynExtraction,
)
from pecos.qeccs.default_qecc import DefaultQECC
from pecos.typing import QECCParams


class Color488(DefaultQECC):
    """Canonical triangular color-code on a 4.8.8 lattice."""

    def __init__(self, distance: int | None = None, **qecc_params: QECCParams) -> None:
        """Initialize the Color 4.8.8 quantum error correcting code.

        Args:
            distance: The code distance. If not provided, must be specified in qecc_params.
            **qecc_params: Additional QECC parameters including:
                - distance: The code distance (required if not provided as first argument)
                - circuit_compiler: The circuit compiler to use (default: OneAncillaPerCheck)
                - mapping: Optional qubit mapping

        Raises:
            Exception: If the distance is even (this code requires odd distance).
        """
        # TODO: Need to switch to codes each having a class defining how classes are implemented. From that we get the
        # layout and ancillas. We don't need a general circuit conversion script... The default implementation may
        # still be overridden.
        # TODO: A layout with syndomes should be created and the implementation should use this is create a physical
        # layout. -> tanner_graph, layout

        if distance is not None:
            qecc_params["distance"] = distance

        super().__init__(**qecc_params)

        # Give name for others classes to identify this code
        # --------------------------------------------------
        self.name = "4.8.8 Color Code"

        # QECC parameters:
        # ----------------
        self.qecc_params = qecc_params

        # Create dictionaries to associate symbols to gate and instruction classes.
        self.sym2gate_class, self.sym2instruction_class = self._set_symbols()

        # d - distance
        self.distance = qecc_params["distance"]

        if self.distance % 2 == 0:
            msg = "This code requires an odd distance!"
            raise Exception(msg)

        # n - number of data qubits
        self.num_data_qudits = int((self.distance**2 - 1) * 0.5 + self.distance)

        # k - number of logical qubits
        self.num_logical_qudits = 1

        # number of syndrome bits
        self.num_syndromes = self.num_data_qudits - self.num_logical_qudits

        # --------------------------------------------------------------------------------------------------------------
        # Determine number of ancillas to reserve given the check circuit implementation and, perhaps, the logical
        # gate circuits implemented by this class.
        # --------------------------------------------------------------------------------------------------------------
        self.circuit_compiler = qecc_params.get(
            "circuit_compiler",
            OneAncillaPerCheck(),
        )
        self.num_ancilla_qudits = self.circuit_compiler.get_num_ancillas(
            self.num_syndromes,
        )

        # Total number of qudits.
        # self.qudit_set, self.data_qudit_set, self.ancilla_qudit_set will be determined when creating the layout.

        # Determine QECC geometry
        # -----------------------
        self.lattice_width = None
        self.lattice_height = None
        self.lattice_dimensions = {}
        self.position2qudit = {}
        self.layout = self._generate_layout()

        # Create side information
        # Allows other classes (e.g., decoders) to understand the orientation of the code
        self.sides = self._determine_sides()

    @staticmethod
    def _set_symbols() -> tuple[dict, dict]:
        # gate and instruction symbol bindings
        # ------------------------------------
        # gate symbol => gate class
        sym2gate_class = {
            "I": GateIdentity,
            "init |0>": GateInitZero,
            "init |+>": GateInitPlus,
        }

        # instruction symbol => instr. class
        sym2instruction_class = {
            "instr_syn_extract": InstrSynExtraction,
            "instr_init_zero": InstrInitZero,
            "instr_init_plus": InstrInitPlus,
        }

        return sym2gate_class, sym2instruction_class

    @staticmethod
    def _get_distance(params: dict[str, Any]) -> tuple[int, int, int]:
        """Check and set the distance.

        :return:
        """
        distance = params.get("distance")

        if distance is not None:
            distance_width = distance_height = distance

        else:
            distance_width = params.get(
                "distance_width",
            )  # The width of the code. == Z? distance
            distance_height = params.get(
                "distance_height",
            )  # The height of the code. == X? distance
            distance = min(distance_width, distance_height)

        return distance, distance_height, distance_width

    def _generate_layout(self) -> dict:
        """Creates the layout dictionary which describes the location of the qubits in the code.

        Uses pecos.qec.color.geometry for data qubit layout generation.
        """
        self.lattice_height = 4 * self.distance - 4
        self.lattice_width = 2 * self.distance - 2

        self.lattice_dimensions = {
            "width": self.lattice_width,
            "height": self.lattice_height,
        }

        # Use pecos.qec for data qubit layout
        nodeid2pos, _ = generate_488_layout(self.distance)

        # Add data qubits to layout
        for nid, pos in nodeid2pos.items():
            self.layout[nid] = pos
            self.position2qudit[pos] = nid
            self.qudit_set.add(nid)
            self.data_qudit_set.add(nid)

        # Add ancilla qubits (check positions)
        ancilla_ids = self._ancilla_id_iter()
        for y in range(self.lattice_width + 1):
            for x in range(self.lattice_height + 1):
                # Skip positions outside the triangular region
                if ((x, y) == (x, x + 2) and x % 2 == 1 and y % 8 == 3) or (
                    (x, y) == (4 * self.distance - y, y) and x % 2 == 1 and y % 8 == 7
                ):
                    pass
                elif (x, y) > (x, x) or (x, y) > (4 * self.distance - y - 2, y):
                    continue

                # Ancilla positions
                if x % 4 == 1 and y % 4 == 3:
                    self._add_node(x, y, ancilla_ids)

                if y == 0 and x % 8 == 5:
                    self._add_node(x, y, ancilla_ids)

        return self.layout

    def _determine_sides(self) -> dict[str, list]:
        """Outputs a dictionary that describes the sides of the code.

        The repetition code is essentially a line.

        d d d
         a a

        d = data
        a = ancilla

        :return:
        """
        bottom_nodes = []
        left_nodes = []

        # self.qubits_data is not set when this is called

        for d, (x, y) in self.layout.items():
            if (x, y) == (x, x):
                left_nodes.append(d)

            # if x == 555:

            if y == 0 and x % 2 == 0:
                bottom_nodes.append(d)

        bottom_nodes.sort()
        left_nodes.sort()

        bottom_nodes = [self.mapping[i] for i in bottom_nodes]
        left_nodes = [self.mapping[i] for i in left_nodes]

        return {
            "bottom": bottom_nodes,
            "left": left_nodes,
        }
