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

"""Rotated Surface 4.4.4.4 quantum error correcting code implementation.

This module provides the Rotated Surface code on a 4.4.4.4 lattice structure,
a topological quantum error correcting code. The rotated layout is the most
commonly used surface code variant.

Note: This is also known as the "medial" surface code in some literature.
"""

from pecos.circuit_converters.checks2circuit import Check2Circuits
from pecos.qec.surface.layouts import generate_surface_layout
from pecos.qeccs.default_qecc import DefaultQECC
from pecos.qeccs.surface_medial_4444.gates import (
    GateIdentity,
    GateInitPlus,
    GateInitZero,
)
from pecos.qeccs.surface_medial_4444.instructions import (
    InstrInitPlus,
    InstrInitZero,
    InstrSynExtraction,
)
from pecos.typing import QECCParams


class SurfaceMedial4444(DefaultQECC):
    """Medial Surface code on 4.4.4.4 lattice."""

    def __init__(
        self,
        distance: int | None = None,
        height: int | None = None,
        width: int | None = None,
        **qecc_params: QECCParams,
    ) -> None:
        """Initialize the SurfaceMedial4444 code with the given parameters.

        Args:
        ----
            distance: The distance of the code. If specified a square code of height and width equaled to the distance
            will be returned.
            height: The height of the code block. This is the size of the minimum logical X.
            width: The width of the code block. This is the size of the minimum logical Z.
            **qecc_params: Additional keyword arguments for QECC configuration
        """
        qecc_params["distance"] = distance
        qecc_params["height"] = height
        qecc_params["width"] = width

        super().__init__(**qecc_params)

        self.rotated = qecc_params.get("rotated", False)

        # Give name for others classes to identify this code
        # --------------------------------------------------
        self.name = "Medial 4.4.4.4 Surface Code"

        # QECC parameters:
        # ----------------
        self.qecc_params = qecc_params

        # Create dictionaries to associate symbols to gate and instruction classes.
        self.sym2gate_class, self.sym2instruction_class = self._set_symbols()

        # d - distance
        # Both gets and sets distance, height, and width in qecc_params
        self.distance, self.height, self.width = self._get_distance()

        # n - number of data qubits
        self.num_data_qudits = self.height * self.width

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

    def _get_distance(self) -> tuple[int, int, int]:
        """Sets the distances based on the `qecc_params` `distance`, `height`, and `width`.

        This will modify the `gate_params`.
        """
        params = self.qecc_params

        distance = params.get("distance")
        width = params.get("width")
        height = params.get("height")

        if width is not None and height is not None:
            if distance is not None:
                msg = "The distance should not be specified if the height and width are."
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

    def _generate_layout(self) -> dict:
        """Creates the layout dictionary which describes the location of the qubits in the code.

        Uses pecos.qec.surface.layouts for rotated surface code layout generation.
        """
        height = self.height
        width = self.width
        lattice_height = 2 * height
        lattice_width = 2 * width
        self.lattice_height = lattice_height
        self.lattice_width = lattice_width

        self.lattice_dimensions = {
            "width": lattice_width,
            "height": lattice_height,
        }

        # Use pecos.qec for layout generation
        data_positions, ancilla_positions = generate_surface_layout(width, height)

        # Add data qubits
        for nid, pos in enumerate(data_positions):
            self.layout[nid] = pos
            self.position2qudit[pos] = nid
            self.qudit_set.add(nid)
            self.data_qudit_set.add(nid)

        # Add ancilla qubits
        ancilla_start_id = len(data_positions)
        for i, pos in enumerate(ancilla_positions):
            nid = ancilla_start_id + i
            self.layout[nid] = pos
            self.position2qudit[pos] = nid
            self.qudit_set.add(nid)
            self.ancilla_qudit_set.add(nid)

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
        width = self.lattice_width
        height = self.lattice_height

        # Logical X
        top_nodes = []
        right_nodes = []
        bottom_nodes = []
        left_nodes = []

        # self.qubits_data is not set when this is called

        for d, (x, y) in self.layout.items():
            if x == 1 and y % 2 == 1:
                left_nodes.append(d)

            if x == width - 1 and y % 2 == 1:
                right_nodes.append(d)

            if y == 1 and x % 2 == 1:
                top_nodes.append(d)

            if y == height - 1 and x % 2 == 1:
                bottom_nodes.append(d)

        top_nodes.sort(reverse=True)
        right_nodes.sort(reverse=True)
        bottom_nodes.sort()
        left_nodes.sort()

        top_nodes = [self.mapping[i] for i in top_nodes]
        right_nodes = [self.mapping[i] for i in right_nodes]
        bottom_nodes = [self.mapping[i] for i in bottom_nodes]
        left_nodes = [self.mapping[i] for i in left_nodes]

        return {
            "top": top_nodes,
            "right": right_nodes,
            "bottom": bottom_nodes,
            "left": left_nodes,
        }
