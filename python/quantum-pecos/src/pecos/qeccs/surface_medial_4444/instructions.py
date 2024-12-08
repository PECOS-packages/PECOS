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

        qecc_init_ticks = qecc.qecc_params.get("init_ticks", 0)
        qecc_meas_ticks = qecc.qecc_params.get("meas_ticks", 7)
        qecc_x_ticks = qecc.qecc_params.get("x_ticks", [2, 4, 3, 5])
        qecc_z_ticks = qecc.qecc_params.get("z_ticks", [2, 4, 3, 5])

        self.init_ticks = params.get("init_ticks", qecc_init_ticks)
        self.meas_ticks = params.get("meas_ticks", qecc_meas_ticks)
        self.x_ticks = params.get("x_ticks", qecc_x_ticks)
        self.z_ticks = params.get("z_ticks", qecc_z_ticks)

        self.abstract_circuit = QuantumCircuit(**params)

        self.data_qudit_set = self.qecc.data_qudit_set
        self.ancilla_qudit_set = self.qecc.ancilla_qudit_set

        self.ancilla_x_check = set()
        self.ancilla_z_check = set()

        # Go through the ancillas and grab the data qubits that are on either side of it.
        layout = qecc.layout  # qudit_id => (x, y)

        self.pos2qudit = pos2qudit(layout)

        for q, (x, y) in layout.items():
            if x % 2 == 0 and y % 2 == 0:
                # Ancilla
                if x % 4 == y % 4:
                    # X check
                    self._create_x_check(q, x, y)

                else:
                    # Z check
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

        self._stabs_destabs = {}

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
            ticks=self.x_ticks,
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
            ticks=self.z_ticks,
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
        """
        From the positions given for possible data qudits, add the qudits and their corresponding ticks for each qudit
        that does exist.

        Args:
            position_to_qudit:
            positions:
            ticks:

        Returns:

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
            (x - 1, y + 1),
            (x + 1, y + 1),
            (x - 1, y - 1),
            (x + 1, y - 1),
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
            (x - 1, y + 1),
            (x - 1, y - 1),
            (x + 1, y + 1),
            (x + 1, y - 1),
        ]

    @property
    def stabs_destabs(self):
        if self._stabs_destabs:
            return self._stabs_destabs

        if self.qecc.height != self.qecc.width:
            msg = "This currently only works for square code blocks."
            raise Exception(msg)

        instr = self

        stabs_row_x = []
        stabs_row_z = []
        destabs_row_x = []
        destabs_row_z = []

        for a in self.ancilla_qudit_set:
            stabs_row_z.append({a})
            stabs_row_x.append(set())
            destabs_row_x.append({a})
            destabs_row_z.append(set())

        xdestabs = self.generate_xdestabs()
        zdestabs = self.generate_zdestabs()

        # Creating stabilizers
        for check_type, _, params in instr.abstract_circuit.items():
            if check_type == "X check":
                # Ancillas initialized in |0>
                # Pauli X-type stabilizers
                stabs_row_x.append(set(params["datas"]))
                stabs_row_z.append(set())
                destabs_row_x.append(set())
                destabs_row_z.append(zdestabs[params["ancillas"]])

            else:
                # Ancillas initialized in |0>
                # Pauli Z-type stabilizers
                stabs_row_z.append(set(params["datas"]))
                stabs_row_x.append(set())
                destabs_row_z.append(set())
                destabs_row_x.append(xdestabs[params["ancillas"]])

        output_dict = {
            "stabs_x": stabs_row_x,
            "stabs_z": stabs_row_z,
            "destabs_x": destabs_row_x,
            "destabs_z": destabs_row_z,
        }

        self._stabs_destabs = output_dict

        return output_dict

    def generate_xdestabs(self):
        distance = self.qecc.distance

        # x-type destabilizers

        xdestabs_temp = []
        # going alone the bottom
        b = 1 if distance % 2 == 0 else 2

        for x in range(b, distance, 2):
            temp = []
            y = distance - 1
            for j in range(distance):
                new_point = (x + j, y - j)

                if new_point[1] <= 0:
                    break

                if new_point[0] > distance - 1:
                    break

                temp.append(new_point)

            xdestabs_temp.append(temp)

        # ----------------
        xdestabs = []
        for ds in xdestabs_temp:
            for i in range(len(ds)):
                temp = []
                for j in range(i + 1):
                    temp.append(ds[j])
                xdestabs.append(temp)
        # -----------------

        # ladder climb
        ladder = []
        x = 0
        for y in range(distance - 1, 0, -1):
            ladder.append((x, y))

        for i in range(len(ladder)):
            xdestabs.append(ladder[: i + 1])

        ladder_points = []
        for i in range((distance + 1) % 2, distance - 1, 2):
            ladder_points.append(i)

        ladder_temp = []
        for i in ladder_points:
            temp = list(ladder[: i + 1])
            x, y = ladder[i]

            for j in range(1, distance):
                if j != 1:
                    temp = list(ladder_temp[-1])
                new_point = (x + j, y - j)

                if new_point[1] <= 0:
                    break

                if new_point[0] >= distance - 1:
                    break

                temp.append(new_point)
                ladder_temp.append(temp)

        xdestabs.extend(ladder_temp)

        set_destabs = {}
        relayout = {v: k for k, v in self.qecc.layout.items()}

        for d in xdestabs:
            row = set()

            # Find the associated ancilla location
            x, y = d[-1]
            a = relayout[(2 * x + 1 + 1, 2 * y + 1 - 1)]

            if a in self.ancilla_x_check:
                a = relayout[(2 * x - 1 + 1, 2 * y + 1 - 1)]

            for x, y in d:
                row.add(relayout[(2 * x + 1, 2 * y + 1)])
            set_destabs[a] = set(row)

        return set_destabs

    def generate_zdestabs(self):
        distance = self.qecc.distance

        # x-type destabilizers

        zdestabs_temp = []
        # going alone the bottom
        b = 2 if distance % 2 == 0 else 1

        for y in range(b, distance, 2):
            temp = []
            x = distance - 1
            for j in range(distance):
                new_point = (x - j, y + j)

                if new_point[0] <= 0:
                    break

                if new_point[1] > distance - 1:
                    break

                temp.append(new_point)

            zdestabs_temp.append(temp)

        # ----------------
        zdestabs = []
        for ds in zdestabs_temp:
            for i in range(len(ds)):
                temp = []
                for j in range(i + 1):
                    temp.append(ds[j])
                zdestabs.append(temp)
        # -----------------

        # ladder climb
        ladder = []
        y = 0
        for x in range(distance - 1, 0, -1):
            ladder.append((x, y))

        for i in range(len(ladder)):
            zdestabs.append(ladder[: i + 1])

        ladder_points = []
        for i in range(distance % 2, distance - 1, 2):
            ladder_points.append(i)

        ladder_temp = []
        for i in ladder_points:
            temp = list(ladder[: i + 1])
            x, y = ladder[i]

            for j in range(1, distance):
                if j != 1:
                    temp = list(ladder_temp[-1])
                new_point = (x - j, y + j)

                if new_point[0] <= 0:
                    break

                if new_point[1] >= distance - 1:
                    break

                temp.append(new_point)
                ladder_temp.append(temp)

        zdestabs.extend(ladder_temp)

        set_destabs = {}
        relayout = {v: k for k, v in self.qecc.layout.items()}

        for d in zdestabs:
            row = set()

            # Find the associated ancilla location
            x, y = d[-1]
            a = relayout[(2 * x + 1 - 1, 2 * y + 1 + 1)]

            if a in self.ancilla_z_check:
                a = relayout[(2 * x + 1 - 1, 2 * y + 1 - 1)]

            for x, y in d:
                row.add(relayout[(2 * x + 1, 2 * y + 1)])
            set_destabs[a] = row

        return set_destabs


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
        self.abstract_circuit.params.update(params)

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

        self._stabs_destabs = {}

    @property
    def stabs_destabs(self):
        if self._stabs_destabs:
            return self._stabs_destabs

        params = self.params
        syn_ext = self.qecc.instruction("instr_syn_extract", **params)

        for name, rows in syn_ext.stabs_destabs.items():
            self._stabs_destabs[name] = []
            for row in rows:
                self._stabs_destabs[name].append(set(row))

        # |0> -> logical Z is a stabilizer
        self._stabs_destabs["stabs_z"].append(set(self.qecc.sides["top"]))
        self._stabs_destabs["stabs_x"].append(set())
        self._stabs_destabs["destabs_x"].append(set(self.qecc.sides["left"]))
        self._stabs_destabs["destabs_z"].append(set())

        return self._stabs_destabs


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

        # This is basically syndrome extraction round where all the data qubits are initialized to plus.
        syn_ext = qecc.instruction("instr_syn_extract", **params)

        # Make a shallow copy of the abstract circuits.
        self.abstract_circuit = syn_ext.abstract_circuit.copy()
        self.abstract_circuit.params.update(params)

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

        self._stabs_destabs = {}

    @property
    def stabs_destabs(self):
        if self._stabs_destabs:
            return self._stabs_destabs

        params = self.params
        syn_ext = self.qecc.instruction("instr_syn_extract", **params)

        for name, rows in syn_ext.stabs_destabs.items():
            self._stabs_destabs[name] = []
            for row in rows:
                self._stabs_destabs[name].append(set(row))

        # |0> -> logical Z is a stabilizer
        self._stabs_destabs["stabs_x"].append(set(self.qecc.sides["left"]))
        self._stabs_destabs["stabs_z"].append(set())
        self._stabs_destabs["destabs_z"].append(set(self.qecc.sides["top"]))
        self._stabs_destabs["stabs_x"].append(set())

        return self._stabs_destabs
