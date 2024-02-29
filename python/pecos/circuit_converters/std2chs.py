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

from pecos.circuits import QuantumCircuit

conversion_dict_one = {
    # Initialization
    # ==============
    "init |0>": ("init |0>",),
    "init |1>": ("init |0>", "H", "S", "S", "H"),  # X |0>
    "init |+>": ("init |0>", "H"),
    "init |->": ("init |0>", "H", "S", "S"),  # ZH |0>
    "init |+i>": ("init |0>", "S", "S", "S", "H", "S"),
    "init |-i>": ("init |0>", "S", "H", "S", "S", "S"),
    # one-qubit operations
    # ====================
    # Paulis
    "I": ("I",),
    "X": ("H", "S", "S", "H"),
    "Y": ("S", "S", "H", "S", "S", "H"),  # Y = iXZ -> Z, X
    "Z": ("S", "S"),
    # Square root of Paulis
    "Q": ("H", "S", "H"),
    "Qd": ("H", "S", "S", "S", "H"),
    "R": ("S", "S", "H"),  # R = HSS
    "Rd": ("H", "S", "S"),  # Rd = SSH
    "S": ("S",),
    "Sd": ("S", "S", "S"),
    # Hadamard-like
    "H": ("H",),
    "H1": ("H",),
    "H2": ("S", "S", "H", "S", "S"),
    "H3": ("H", "S", "S", "H", "S"),
    "H4": ("H", "S", "S", "H", "S", "S", "S"),
    "H5": ("S", "S", "S", "H", "S"),
    "H6": ("S", "H", "S", "S", "S"),
    "H+z+x": ("H",),
    "H-z-x": ("S", "S", "H", "S", "S"),
    "H+y-z": ("H", "S", "S", "H", "S"),
    "H-y-z": ("H", "S", "S", "H", "S", "S", "S"),
    "H-x+y": ("S", "S", "S", "H", "S"),
    "H-x-y": ("S", "H", "S", "S", "S"),
    # Face rotations
    "F1": ("S", "S", "S", "H"),  # HSSS
    "F1d": ("H", "S"),
    "F2": ("S", "S", "H", "S"),
    "F2d": ("S", "S", "S", "H", "S", "S"),
    "F3": ("S", "H", "S", "S"),
    "F3d": ("S", "S", "H", "S", "S", "S"),
    "F4": ("H", "S", "S", "S"),
    "F4d": ("S", "H"),
    # Measurements
    # ============
    "measure X": ("H", "measure Z", "H"),
    "measure Y": ("S", "S", "S", "H", "S", "measure Z", "S", "S", "S", "H", "S"),
    "measure Z": ("measure Z",),
    "force output": "force output",
}

conversion_dict_two = {
    # two-qubit operations
    # ====================
    "CNOT": ("CNOT",),
    "CZ": (("I", "H"), "CNOT", ("I", "H")),
    "SWAP": ("CNOT", "TONC", "CNOT"),
    "G": (("I", "H"), "CNOT", ("H", "H"), "CNOT", ("I", "H")),
    "II": (("I", "I"),),
}

measurements = {"measure X", "measure Y", "measure Z"}


def std2chs(quantum_circuit):
    """Used to convert a QuantumCircuit to one using a description of only H, S, and CNOT.

    Args:
        quantum_circuit(QuantumCircuit):

    Returns:

    """
    compiled_circuit = QuantumCircuit()

    for symbol, locations, params in quantum_circuit.items(params=True):
        # Convert symbols that act on single qubits
        if symbol in conversion_dict_one:
            new_symbols = conversion_dict_one[symbol]

            for sym in new_symbols:
                if sym in measurements:
                    compiled_circuit.append(sym, locations, **params)
                else:
                    compiled_circuit.append(sym, locations)

        # Convert symbols that act on two qubits
        elif symbol in conversion_dict_two:
            new_symbols = conversion_dict_two[symbol]
            first_locations = set()
            second_locations = set()
            for loc in locations:
                q1, q2 = loc
                first_locations.add(q1)
                second_locations.add(q2)

            for sym in new_symbols:
                if isinstance(sym, tuple):  # Collection of single qubits
                    sym1, sym2 = sym

                    if sym1 != "I":
                        if sym1 in measurements:
                            compiled_circuit.append(sym1, first_locations, **params)
                        else:
                            compiled_circuit.append(sym1, first_locations)

                    if sym2 != "I":
                        if sym2 in measurements:
                            compiled_circuit.append(sym2, second_locations, **params)
                        else:
                            compiled_circuit.append(sym2, second_locations)
                elif sym != "TONC":
                    compiled_circuit.append(sym, locations, **params)
                else:
                    new_locations = set()

                    for loc in locations:
                        qc, qt = loc
                        new_locations.add((qt, qc))

                    compiled_circuit.append("CNOT", new_locations, **params)

        else:
            raise Exception(
                'Symbol "%s" is not currently handled by this converter!' % symbol,
            )

    return compiled_circuit
