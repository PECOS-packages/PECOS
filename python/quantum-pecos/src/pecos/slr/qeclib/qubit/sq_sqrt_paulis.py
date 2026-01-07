"""Single-qubit square root Pauli gate implementations.

This module provides square root of Pauli gate implementations (√X, √Y, √Z)
which are Clifford gates that when applied twice give the corresponding
Pauli operators, useful in quantum error correction protocols.
"""

# Copyright 2024 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from pecos.slr.qeclib.qubit.qgate_base import QGate


class SX(QGate):
    """Square root of Pauli X gate.

    Action on Pauli operators:
    X -> X
    Z -> -Y
    Y -> Z
    """


class SY(QGate):
    """Square root of Pauli Y gate.

    This gate is the square root of the Pauli Y operation.
    """


class SZ(QGate):
    """Square root of Pauli Z gate (S gate).

    This gate performs a π/2 rotation around the Z-axis.
    """


S = SZ


class SXdg(QGate):
    """Inverse square root of Pauli X gate.

    This gate is the inverse of the square root of Pauli X operation.
    """


class SYdg(QGate):
    """Inverse square root of Pauli Y gate.

    This gate is the inverse of the square root of Pauli Y operation.
    """


class SZdg(QGate):
    """Inverse square root of Pauli Z gate (S-dagger gate).

    This gate performs a -π/2 rotation around the Z-axis.
    """


Sdg = SZdg
