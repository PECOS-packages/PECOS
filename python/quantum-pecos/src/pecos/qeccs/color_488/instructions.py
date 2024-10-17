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

        self.abstract_circuit = QuantumCircuit(**params)

        self.ancilla_x_check = set()
        self.ancilla_z_check = set()

        # Go through the ancillas and grab the data qubits that are on either side of it.

        self.pos2qudit = pos2qudit(qecc.layout)

        for q in sorted(self.ancilla_qudit_set):
            self._create_checks(q)

        # Determine the logical operations
        # --------------------------------
        z_qudits = set(qecc.sides["bottom"])
        x_qudits = set(qecc.sides["bottom"])

        logical_ops = [  # Each element in the list corresponds to a logical qubit
            # The keys label the type of logical operator
            {
                "X": QuantumCircuit([{"X": x_qudits}]),
                "Z": QuantumCircuit([{"Z": z_qudits}]),
            },
        ]

        self.initial_logical_ops = logical_ops
        self.final_logical_ops = logical_ops

        self.logical_signs = None
        self.logical_stabilizers = None

        # Must be called at the end of initiation.
        self._compile_circuit(self.abstract_circuit)

    def _create_checks(self, ancilla):
        self.ancilla_x_check.add(ancilla)
        self.ancilla_z_check.add(ancilla)

        # look up ancilla location
        x, y = self.qecc.layout[ancilla]

        square = []
        octagon = []

        square.extend(
            (
                self.pos2qudit.get((x - 1, y + 1)),
                self.pos2qudit.get((x + 1, y + 1)),
                self.pos2qudit.get((x - 1, y - 1)),
                self.pos2qudit.get((x + 1, y - 1)),
            ),
        )

        found_square = False

        for d in square:
            if d is not None:
                found_square = True
                break

        if found_square:
            self.abstract_circuit.append(
                "X check",
                polygon="square",
                locations={ancilla},
                datas=square,
            )
            self.abstract_circuit.append(
                "Z check",
                polygon="square",
                locations={ancilla},
                datas=square,
            )

        else:
            if y != 0:
                octagon.extend(
                    (
                        self.pos2qudit.get((x - 1, y + 3)),
                        self.pos2qudit.get((x + 1, y + 3)),
                        self.pos2qudit.get((x - 3, y + 1)),
                        self.pos2qudit.get((x + 3, y + 1)),
                        self.pos2qudit.get((x - 3, y - 1)),
                        self.pos2qudit.get((x + 3, y - 1)),
                        self.pos2qudit.get((x - 1, y - 3)),
                        self.pos2qudit.get((x + 1, y - 3)),
                    ),
                )
            else:
                octagon.extend(
                    (
                        self.pos2qudit.get((x - 1, y + 2)),
                        self.pos2qudit.get((x + 1, y + 2)),
                        self.pos2qudit.get((x - 3, y)),
                        self.pos2qudit.get((x + 3, y)),
                    ),
                )
                octagon.extend([None, None, None, None])

            self.abstract_circuit.append(
                "X check",
                polygon="octagon",
                locations={ancilla},
                datas=octagon,
            )
            self.abstract_circuit.append(
                "Z check",
                polygon="octagon",
                locations={ancilla},
                datas=octagon,
            )


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
                "X": QuantumCircuit([{"Z": set(qecc.sides["bottom"])}]),
                "Z": QuantumCircuit([{"X": set(qecc.sides["bottom"])}]),
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
                "X": QuantumCircuit([{"X": set(qecc.sides["bottom"])}]),
                "Z": QuantumCircuit([{"Z": set(qecc.sides["bottom"])}]),
            },
        ]

        # List of corresponding logical sign. (The logical sign if the instruction is preformed ideally.)
        self.logical_signs = [0]
        self.logical_stabilizers = ["X"]
        # ---------------------------------

        # Must be called at the end of initiation.
        self._compile_circuit(self.abstract_circuit)
