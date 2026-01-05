"""Logical Hadamard gates for the Steane 7-qubit code.

This module provides logical Hadamard gate implementations for the Steane 7-qubit code, performing transversal
operations that preserve the error correction properties of the code.
"""

# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from pecos.slr import Block, QReg
from pecos.slr.qeclib import qubit as qb


class SZ(Block):
    """Square root of Z gate (S gate).

    Also known as the S gate with matrix representation diag(1, i).

    Action on Pauli operators:
    X -> Y
    Z -> Z
    Y -> -X
    """

    def __init__(self, q: QReg) -> None:
        """Initialize a logical SZ gate on the color code.

        Args:
            q: A quantum register containing qubits representing a logical qubit in the color code.
        """
        # TODO: Verify if the physical implementation of the S gate alternates per distance...

        super().__init__(
            qb.SZdg(q),
        )


class SZdg(Block):
    """Hermitian adjoint of the square root of Z gate (S† gate).

    Also known as the Sdg gate with matrix representation diag(1, -i).

    Action on Pauli operators:
    X -> -Y
    Z -> Z
    Y -> X
    """

    def __init__(self, q: QReg) -> None:
        """Initialize a logical SZdg gate on the color code.

        Args:
            q: A quantum register containing qubits representing a logical qubit in the color code.
        """
        # TODO: Verify if the physical implementation of the Sdg gate alternates per distance...

        super().__init__(
            qb.SZ(q),
        )
