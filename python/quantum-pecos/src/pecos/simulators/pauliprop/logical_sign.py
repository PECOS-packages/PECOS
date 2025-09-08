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

"""Logical sign tracking for Pauli fault propagation simulator.

This module provides logical sign tracking functionality for the Pauli fault propagation simulator, managing the
global phase and logical operator signs that arise from Pauli frame propagation in stabilizer circuits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos.circuits import QuantumCircuit
    from pecos.simulators.pauliprop.state import PauliProp


def find_logical_signs(state: PauliProp, logical_circuit: QuantumCircuit) -> int:
    """Find the sign of the logical operator.

    Args:
        state: The PauliProp state instance.
        logical_circuit (QuantumCircuit): The logical circuit to find the sign of.
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
            msg = f'Can not currently handle logical operator with operator "{symbol}"!'
            raise Exception(
                msg,
            )

    anticom = len(state.faults["X"] & logical_zs)
    anticom += len(state.faults["Y"] & logical_zs)
    anticom += len(state.faults["Y"] & logical_xs)
    anticom += len(state.faults["Z"] & logical_xs)

    return anticom % 2
