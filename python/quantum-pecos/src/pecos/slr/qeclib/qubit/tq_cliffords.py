"""Two-qubit Clifford gate implementations.

This module provides two-qubit Clifford gate implementations including
controlled gates (CNOT, CZ) that preserve the Clifford group and are
fundamental to stabilizer-based quantum error correction.
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

from pecos.slr.qeclib.qubit.qgate_base import TQGate


class CX(TQGate):
    """Controlled-X (CNOT) gate.

    This gate flips the target qubit if the control qubit is in state |1⟩.
    """


class CY(TQGate):
    """Controlled-Y gate.

    This gate applies a Pauli-Y to the target qubit if the control qubit is in state |1⟩.
    """


class CZ(TQGate):
    """Controlled-Z gate.

    This gate applies a Pauli-Z to the target qubit if the control qubit is in state |1⟩.
    """


class SXX(TQGate):
    """Two-qubit square root of XX interaction gate.

    This gate implements the Sqrt XX interaction between two qubits.
    """


class SYY(TQGate):
    """Two-qubit square root of YY interaction gate.

    This gate implements the Sqrt YY interaction between two qubits.
    """


class SZZ(TQGate):
    """Two-qubit square root of ZZ interaction gate.

    This gate implements the Sqrt ZZ interaction between two qubits.
    """


class SXXdg(TQGate):
    """Inverse two-qubit square root of XX interaction gate.

    This gate implements the inverse of the Sqrt XX interaction between two qubits.
    """


class SYYdg(TQGate):
    """Inverse two-qubit square root of YY interaction gate.

    This gate implements the inverse of the Sqrt YY interaction between two qubits.
    """


class SZZdg(TQGate):
    """Inverse two-qubit square root of ZZ interaction gate.

    This gate implements the inverse of the Sqrt ZZ interaction between two qubits.
    """
