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

"""repetition_z
~~~~~~~~~~~~.

Generates circuits for the repetition code in the Z-Basis.
"""

from pecos.circuit_converters.checks2circuit import Check2Circuits
from pecos.qeccs.qecc_parent_class import QECC
from pecos.qeccs.surface_4444.gates import GateIdentity, GateInitPlus, GateInitZero
from pecos.qeccs.surface_4444.instructions import (
    InstrInitPlus,
    InstrInitZero,
    InstrSynExtraction,
)


class Surface4444(QECC):
    """Non-medial Surface code on 4.4.4.4 lattice."""

    def __init__(self, distance=None, height=None, width=None, **qecc_params) -> None:
        """Args:
        ----
            distance: The distance of the code. If specified a square code of height and width equaled to the distance
            will be returned.
            height: The height of the code block. This is the size of the minimum logical X.
            width: The width of the code block. This is the size of the minimum logical Z.
            **qecc_params:
        """
        qecc_params["distance"] = distance
        qecc_params["height"] = height
        qecc_params["width"] = width

        super().__init__(**qecc_params)

        # Give name for others classes to identify this code
        # --------------------------------------------------
        self.name = "4.4.4.4 Surface Code"

        # QECC parameters:
        # ----------------
        self.qecc_params = qecc_params

        # Create dictionaries to associate symbols to gate and instruction classes.
        self.sym2gate_class, self.sym2instruction_class = self._set_symbols()

        # d - distance
        # Both gets and sets distance, height, and width in qecc_params
        self.distance, self.height, self.width = self._get_distance()

        # n - number of data qubits
        self.num_data_qudits = (
            2 * self.height * self.width - self.height - self.width + 1
        )

        # k - number of logical qubits
        self.num_logical_qudits = 1

        # number of syndrome bits
        self.num_syndromes = self.num_data_qudits - self.num_logical_qudits

        # --------------------------------------------------------------------------------------------------------------
        # Determine number of ancillas to reserve given the check circuit implementation and, perhaps, the logical
        # gate circuits implemented by this class.
        # --------------------------------------------------------------------------------------------------------------
        self.circuit_compiler = qecc_params.get("circuit_compiler", Check2Circuits())
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
        self.position_to_qubit = {}
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

    def _get_distance(self):
        """Check and set the distance
        :return:
        """
        params = self.qecc_params

        distance = params.get("distance")
        width = params.get("width")
        height = params.get("height")

        if width is not None and height is not None:
            if distance is not None:
                msg = (
                    "The distance should not be specified if the height and width are."
                )
                raise Exception(msg)

            distance = min(width, height)

        elif distance is not None:
            if width is not None or height is not None:
                msg = "If the distance is specified then neither the height or the width should be."
                raise Exception(msg)

            width = height = distance

        else:
            msg = "Either distance or both height and width should be specified."
            raise Exception(msg)

        self.qecc_params["distance"] = distance
        self.qecc_params["height"] = height
        self.qecc_params["width"] = width

        return distance, height, width

    def _data_id_iter(self):
        """Assigns qudit ids. Also, records qudit id in the sets self.

        Returns:
        -------

        """
        while True:
            qudit_id = max(self.qudit_set, default=-1) + 1
            self.qudit_set.add(qudit_id)
            self.data_qudit_set.add(qudit_id)

            if len(self.data_qudit_set) > self.num_data_qudits:
                msg = "Number of data qudits requested exceeds number expected."
                raise Exception(msg)

            yield qudit_id

    def _ancilla_id_iter(self):
        """Assigns qudit ids. Also, records qudit id in the sets self.

        Returns:
        -------

        """
        last_ancilla_id = None

        while True:
            if len(self.ancilla_qudit_set) == self.num_ancilla_qudits:
                print("Requesting more qudits then expected assuming last ancilla id.")
                yield last_ancilla_id
            else:
                qudit_id = max(self.qudit_set, default=-1) + 1
                last_ancilla_id = qudit_id
                self.qudit_set.add(qudit_id)
                self.ancilla_qudit_set.add(qudit_id)

                yield qudit_id

    def _add_node(self, x, y, iter_ids):
        nid = next(iter_ids)

        self.layout[nid] = (x, y)
        self.position_to_qubit[(x, y)] = nid

    def _generate_layout(self):
        """Creates the layout dictionary which describes the location of the qubits in the code.

        :param qudit_ids:
        :return:
        """
        height = self.height
        width = self.width
        lattice_height = 2 * (height - 1)
        lattice_width = 2 * (width - 1)
        self.lattice_height = lattice_height
        self.lattice_width = lattice_width
        data_ids = self._data_id_iter()
        ancilla_ids = self._ancilla_id_iter()

        self.lattice_dimensions = {
            "width": lattice_width,
            "height": lattice_width,
        }

        # Determine the position of things
        for y in range(lattice_height + 1):
            for x in range(lattice_width + 1):
                if (x % 2 == 0 and y % 2 == 0) or (x % 2 == 1 and y % 2 == 1):
                    # Data
                    self._add_node(x, y, data_ids)

                elif x % 2 == 1 and y % 2 == 0:
                    # X ancilla
                    self._add_node(x, y, ancilla_ids)

                elif x % 2 == 0 and y % 2 == 1:
                    # Z ancilla
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
        width = self.lattice_width
        height = self.lattice_height

        # Logical X
        top_nodes = []
        right_nodes = []
        bottom_nodes = []
        left_nodes = []

        # self.qubits_data is not set when this is called

        for d, (x, y) in self.layout.items():
            if x == 0 and y % 2 == 0:
                left_nodes.append(d)

            if x == width and y % 2 == 0:
                right_nodes.append(d)

            if y == 0 and x % 2 == 0:
                bottom_nodes.append(d)

            if y == height and x % 2 == 0:
                top_nodes.append(d)

        top_nodes.sort()
        right_nodes.sort(reverse=True)
        bottom_nodes.sort(reverse=True)
        left_nodes.sort()

        return {
            "top": top_nodes,
            "right": right_nodes,
            "bottom": bottom_nodes,
            "left": left_nodes,
        }
