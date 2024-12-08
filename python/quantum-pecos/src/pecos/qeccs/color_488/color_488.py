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

"""repetition_z
~~~~~~~~~~~~.

Generates circuits for the repetition code in the Z-Basis.
"""

from pecos.qeccs.color_488.circuit_implementation1 import OneAncillaPerCheck
from pecos.qeccs.color_488.gates import GateIdentity, GateInitPlus, GateInitZero
from pecos.qeccs.color_488.instructions import (
    InstrInitPlus,
    InstrInitZero,
    InstrSynExtraction,
)
from pecos.qeccs.qecc_parent_class import QECC


class Color488(QECC):
    """Canonical triangular color-code on a 4.8.8 lattice."""

    def __init__(self, distance=None, **qecc_params) -> None:
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
    def _set_symbols():
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
    def _get_distance(params):
        """Check and set the distance
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

    def _generate_layout(self):
        """Creates the layout dictionary which describes the location of the qubits in the code."""
        self.lattice_height = 4 * self.distance - 4
        self.lattice_width = 2 * self.distance - 2
        data_ids = self._data_id_iter()
        ancilla_ids = self._ancilla_id_iter()

        self.lattice_dimensions = {
            "width": self.lattice_width,
            "height": self.lattice_height,
        }

        # Determine the position of things
        for y in range(self.lattice_width + 1):
            for x in range(self.lattice_height + 1):
                if ((x, y) == (x, x + 2) and x % 2 == 1 and y % 8 == 3) or (
                    (x, y) == (4 * self.distance - y, y) and x % 2 == 1 and y % 8 == 7
                ):
                    pass

                elif (x, y) > (x, x) or (x, y) > (4 * self.distance - y - 2, y):
                    continue

                if x % 2 == 0 and y % 2 == 0:  # Data
                    if (y / 2) % 4 == 1 or (y / 2) % 4 == 2:
                        if (x / 2) % 4 == 2 or (x / 2) % 4 == 3:
                            self._add_node(x, y, data_ids)

                    else:
                        if (x / 2) % 4 == 0 or (x / 2) % 4 == 1:
                            self._add_node(x, y, data_ids)

                if x % 4 == 1 and y % 4 == 3:
                    self._add_node(x, y, ancilla_ids)

                if y == 0 and x % 8 == 5:
                    self._add_node(x, y, ancilla_ids)

        return self.layout

    def _determine_sides(self):
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
