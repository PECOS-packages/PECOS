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

"""Used to determine if two operators (circuits) commute.

The main use case for this is to determine if an ideal recovery operator would change a logical sign or not.
"""


def qubit_pauli(first_circuit, second_circuit):
    if len(first_circuit) != 1 or len(second_circuit) != 1:
        msg = "Circuits are expected to only have one tick."
        raise Exception(msg)

    first_xs = set()
    first_zs = set()
    for symbol, gate_locations, _ in first_circuit.items():
        if symbol == "X":
            first_xs.update(gate_locations)
        elif symbol == "Z":
            first_zs.update(gate_locations)
        elif symbol == "Y":
            first_xs.update(gate_locations)
            first_zs.update(gate_locations)
        else:
            raise Exception(
                'Can not currently handle logical operator with operator "%s"!'
                % symbol,
            )

    second_xs = set()
    second_zs = set()
    for symbol, gate_locations, _ in second_circuit.items():
        if symbol == "X":
            second_xs.update(gate_locations)
        elif symbol == "Z":
            second_zs.update(gate_locations)
        elif symbol == "Y":
            second_xs.update(gate_locations)
            second_zs.update(gate_locations)
        else:
            raise Exception(
                'Can not currently handle logical operator with operator "%s"!'
                % symbol,
            )

    return not (len(first_xs & second_zs) + len(first_zs & second_xs)) % 2
