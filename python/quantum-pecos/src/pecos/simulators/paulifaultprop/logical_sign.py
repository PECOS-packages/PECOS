# Copyright 2018 The PECOS Developers
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


def find_logical_signs(state, logical_circuit: QuantumCircuit) -> int:
    """Find the sign of the logical operator.

    Args:
    ----
        state:
        logical_circuit:

    Returns:
    -------

    """
    if len(logical_circuit) != 1:
        msg = "Logical operators are expected to only have one tick."
        raise Exception(msg)

    logical_xs = set()
    logical_zs = set()

    for symbol, gate_locations, _ in logical_circuit.items():
        if symbol == "X":
            logical_xs.update(gate_locations)
        elif symbol == "Z":
            logical_zs.update(gate_locations)
        elif symbol == "Y":
            logical_xs.update(gate_locations)
            logical_zs.update(gate_locations)
        else:
            raise Exception(
                'Can not currently handle logical operator with operator "%s"!'
                % symbol,
            )

    anticom = len(state.faults["X"] & logical_zs)
    anticom += len(state.faults["Y"] & logical_zs)
    anticom += len(state.faults["Y"] & logical_xs)
    anticom += len(state.faults["Z"] & logical_xs)

    return anticom % 2
