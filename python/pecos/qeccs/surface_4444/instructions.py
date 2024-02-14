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

from pecos.circuits.quantum_circuit import QuantumCircuit
from pecos.qeccs.helper_functions import pos2qudit
from pecos.qeccs.instruction_parent_class import LogicalInstruction


class InstrSynExtraction(LogicalInstruction):
    """Instruction for a round of syndrome extraction.

    Parent class sets self.qecc.
    """

    def __init__(self, qecc, symbol, **params) -> None:
        super().__init__(qecc, symbol, **params)

        self.symbol = "instr_syn_extract"

        self.init_ticks = params.get("init_ticks", 0)
        self.meas_ticks = params.get("meas_ticks", 7)
        self.data_ticks = params.get("data_ticks", [2, 4, 3, 5])

        self.abstract_circuit = QuantumCircuit(track_qudits=False, **params)

        self.data_qudit_set = self.qecc.data_qudit_set
        self.ancilla_qudit_set = self.qecc.ancilla_qudit_set

        self.ancilla_x_check = set()
        self.ancilla_z_check = set()

        # Go through the ancillas and grab the data qubits that are on either side of it.
        layout = qecc.layout  # qudit_id => (x, y)

        self.pos2qudit = pos2qudit(layout)

        for q, (x, y) in layout.items():
            if x % 2 == 1 and y % 2 == 0:
                # X ancilla
                self._create_x_check(q, x, y)

            elif x % 2 == 0 and y % 2 == 1:
                # Z ancilla
                self._create_z_check(q, x, y)

        # Determine the logical operations
        # --------------------------------
        z_qudits = set(qecc.sides["top"])
        x_qudits = set(qecc.sides["left"])

        logical_ops = [  # Each element in the list corresponds to a logical qubit
            # The keys label the type of logical operator
            {
                "X": QuantumCircuit([{"X": x_qudits}]),
                "Z": QuantumCircuit([{"Z": z_qudits}]),
            },
        ]

        self.initial_logical_ops = logical_ops

        logical_ops = [  # Each element in the list corresponds to a logical qubit
            # The keys label the type of logical operator
            {
                "X": QuantumCircuit([{"X": x_qudits}]),
                "Z": QuantumCircuit([{"Z": z_qudits}]),
            },
        ]

        self.final_logical_ops = logical_ops

        self.logical_signs = None
        self.logical_stabilizers = None

        # Must be called at the end of initiation.
        self._compile_circuit(self.abstract_circuit)

    def _create_x_check(self, ancilla, x, y):
        """Creates X-checks for circuit_extended."""
        # register the x syndrome ancillas
        self.ancilla_x_check.add(ancilla)

        # get where the position of where the data qubits should be relative to the ancilla
        data_pos = self._data_pos_x_check(x, y)

        # Get the actual, available data-qubits and their ticks that correspond to the possible data qubit positions
        datas, my_data_ticks = self._find_data(
            position_to_qudit=self.pos2qudit,
            positions=data_pos,
            # ticks=self.x_ticks)
            ticks=self.data_ticks,
        )

        # Now add the check to the extended circuit
        locations = set(datas)
        locations.add(ancilla)
        self.abstract_circuit.append(
            "X check",
            locations=locations,
            datas=datas,
            ancillas=ancilla,
            ancilla_ticks=self.init_ticks,
            data_ticks=my_data_ticks,
            meas_ticks=self.meas_ticks,
        )

    def _create_z_check(self, ancilla, x, y):
        """Creates Z-checks for circuit_extended."""
        # register the z syndrome ancillas
        self.ancilla_z_check.add(ancilla)

        # get where the position of where the data qubits should be relative to the ancilla
        data_pos = self._data_pos_z_check(x, y)
        # Get the actual, available data-qubits and their ticks that correspond to the possible data qubit positions
        datas, my_data_ticks = self._find_data(
            position_to_qudit=self.pos2qudit,
            positions=data_pos,
            # ticks=self.z_ticks)
            ticks=self.data_ticks,
        )

        # Now add the check to the extended circuit
        locations = set(datas)
        locations.add(ancilla)
        self.abstract_circuit.append(
            "Z check",
            locations=locations,
            datas=datas,
            ancillas=ancilla,
            ancilla_ticks=self.init_ticks,
            data_ticks=my_data_ticks,
            meas_ticks=self.meas_ticks,
        )

    @staticmethod
    def _find_data(position_to_qudit, positions, ticks):
        """From the positions given for possible data qudits, add the qudits and their corresponding ticks for each
        qudit that does exist.
        """
        data_list = []
        tick_list = []

        for i, p in enumerate(positions):
            data = position_to_qudit.get(p, None)
            if data is not None:
                data_list.append(data)
                tick_list.append(ticks[i])

        return data_list, tick_list

    @staticmethod
    def _data_pos_z_check(x, y):
        """Determines the position of data qudits in a Z check in order of ticks.

        Check direction:   1  |  2
                              |
                           ---+---
                              |
                           3  |  4


        """
        return [
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
            (x + 1, y),
        ]

    @staticmethod
    def _data_pos_x_check(x, y):
        """Determines the position of data qudits in a Z check in order of ticks.

        Check direction:   1  |  3
                              |
                           ---+---
                              |
                           2  |  4
        """
        return [
            (x - 1, y),
            (x, y - 1),
            (x, y + 1),
            (x + 1, y),
        ]


class InstrInitZero(LogicalInstruction):
    """Instruction for initializing a logical zero.

    It is just like syndrome extraction except the data qubits are initialized in the zero state at tick = 0.

    `ideal_meas` == True will cause the measurements to be replace with ideal measurements.

    Parent class sets self.qecc.
    """

    def __init__(self, qecc, symbol, **params) -> None:
        super().__init__(qecc, symbol, **params)

        self.symbol = "instr_init_zero"

        self.data_qudit_set = self.qecc.data_qudit_set
        self.ancilla_qudit_set = self.qecc.ancilla_qudit_set

        # This is basically syndrome extraction round where all the data qubits are initialized to zero.
        syn_ext = qecc.instruction("instr_syn_extract", **params)

        # Make a shallow copy of the abstract circuits.
        self.abstract_circuit = syn_ext.abstract_circuit.copy()
        self.abstract_circuit.metadata.update(params)

        self.ancilla_x_check = syn_ext.ancilla_x_check
        self.ancilla_z_check = syn_ext.ancilla_z_check

        data_qudits = syn_ext.data_qudit_set
        self.abstract_circuit.append("init |0>", locations=data_qudits, tick=0)

        self.initial_logical_ops = [  # Each element in the list corresponds to a logical qubit
            # The keys label the type of logical operator
            {"X": None, "Z": None},  # None => can be anything
        ]

        # Special for state initialization:
        # ---------------------------------
        # list of tuples of logical check and delogical stabilizer for each logical qudit.
        self.final_logical_ops = [
            {
                "Z": QuantumCircuit([{"Z": set(qecc.sides["top"])}]),
                "X": QuantumCircuit([{"X": set(qecc.sides["left"])}]),
            },
        ]

        # List of corresponding logical sign. (The logical sign if the instruction is preformed ideally.)
        self.logical_signs = [0]
        self.logical_stabilizers = ["Z"]
        # ---------------------------------

        # Must be called at the end of initiation.
        self._compile_circuit(self.abstract_circuit)


class InstrInitPlus(LogicalInstruction):
    """Instruction for initializing a logical plus.

    It is just like syndrome extraction except the data qubits are initialized in the plus state at tick = 0.

    `ideal_meas` == True will cause the measurements to be replace with ideal measurements.

    Parent class sets self.qecc.
    """

    def __init__(self, qecc, symbol, **params) -> None:
        super().__init__(qecc, symbol, **params)

        self.symbol = "instr_init_plus"

        self.data_qudit_set = self.qecc.data_qudit_set
        self.ancilla_qudit_set = self.qecc.ancilla_qudit_set

        # This is basically syndrome extraction round where all the data qubits are initialized to zero.
        syn_ext = qecc.instruction("instr_syn_extract", **params)

        # Make a shallow copy of the abstract circuits.
        self.abstract_circuit = syn_ext.abstract_circuit.copy()
        self.abstract_circuit.metadata.update(params)

        self.ancilla_x_check = syn_ext.ancilla_x_check
        self.ancilla_z_check = syn_ext.ancilla_z_check

        data_qudits = syn_ext.data_qudit_set
        self.abstract_circuit.append("init |0>", locations=data_qudits, tick=0)
        self.abstract_circuit.append("H", locations=data_qudits, tick=1)

        self.initial_logical_ops = [  # Each element in the list corresponds to a logical qubit
            # The keys label the type of logical operator
            {"X": None, "Z": None},  # None => can be anything
        ]

        # Special for state initialization:
        # ---------------------------------
        # list of tuples of logical check and delogical stabilizer for each logical qudit.
        self.final_logical_ops = [
            {
                "X": QuantumCircuit([{"X": set(qecc.sides["left"])}]),
                "Z": QuantumCircuit([{"Z": set(qecc.sides["top"])}]),
            },
        ]

        # List of corresponding logical sign. (The logical sign if the instruction is preformed ideally.)
        self.logical_signs = [0]
        self.logical_stabilizers = ["X"]
        # ---------------------------------

        # Must be called at the end of initiation.
        self._compile_circuit(self.abstract_circuit)
