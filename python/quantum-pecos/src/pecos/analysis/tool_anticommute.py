"""Anticommutation analysis tools for quantum circuits.

This module provides utilities for analyzing anticommutation relationships
between quantum circuits and operators, which is essential for understanding
operator algebra in quantum error correction and quantum computing.
"""

# Copyright 2022 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos.circuits import QuantumCircuit


def anticommute(qc1: QuantumCircuit, qc2: QuantumCircuit) -> int:
    """Check if two Pauli operator circuits anticommute.

    Determines whether two quantum circuits representing Pauli operators
    anticommute by counting overlapping qubits with different Pauli operators.

    Args:
        qc1: First quantum circuit containing only Pauli X, Y, Z gates.
        qc2: Second quantum circuit containing only Pauli X, Y, Z gates.

    Returns:
        1 if the circuits anticommute, 0 if they commute.

    Raises:
        Exception: If circuits contain non-Pauli gates.
    """
    x1 = set()
    y1 = set()
    z1 = set()

    x2 = set()
    y2 = set()
    z2 = set()

    for sym, qubits, _ in qc1.items():
        if sym == "X":
            x1 |= qubits
        elif sym == "Y":
            y1 |= qubits
        elif sym == "Z":
            z1 |= qubits
        else:
            msg = "Waaa"
            raise Exception(msg)

    for sym, qubits, _ in qc2.items():
        if sym == "X":
            x2 |= qubits
        elif sym == "Y":
            y2 |= qubits
        elif sym == "Z":
            z2 |= qubits
        else:
            msg = "Waaa"
            raise Exception(msg)

    anticom = len(x1 & z2)
    anticom += len(x1 & y2)

    anticom += len(y1 & z2)
    anticom += len(y1 & x2)

    anticom += len(z1 & x2)
    anticom += len(z1 & y2)

    return anticom % 2
