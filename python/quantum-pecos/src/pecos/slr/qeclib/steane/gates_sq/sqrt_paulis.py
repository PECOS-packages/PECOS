"""Logical square root Pauli gates for the Steane 7-qubit code.

This module provides logical square root Pauli gate implementations for the Steane 7-qubit code, implemented as
transversal Clifford operations that preserve the error correction properties of the code.
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

from pecos.slr import Block, Comment, QReg
from pecos.slr.qeclib import qubit


class SX(Block):
    """Square root of X.

    X -> X
    Z -> -Y

    Y -> Z
    """

    def __init__(self, q: QReg) -> None:
        """Initialize a logical square root of X gate on the Steane code.

        Args:
            q: A quantum register containing exactly 7 qubits representing a logical qubit
                in the Steane code.

        Raises:
            Exception: If the quantum register does not contain exactly 7 qubits.
        """
        if len(q.elems) != 7:
            msg = f"Size of register {len(q.elems)} != 7"
            raise Exception(msg)

        super().__init__(
            Comment("Logical SX"),
            qubit.SXdg(q),
        )


class SXdg(Block):
    """Hermitian adjoint of the square root of X.

    X -> X
    Z -> Y

    Y -> -Z
    """

    def __init__(self, q: QReg) -> None:
        """Initialize a logical adjoint square root of X gate on the Steane code.

        Args:
            q: A quantum register containing exactly 7 qubits representing a logical qubit
                in the Steane code.

        Raises:
            Exception: If the quantum register does not contain exactly 7 qubits.
        """
        if len(q.elems) != 7:
            msg = f"Size of register {len(q.elems)} != 7"
            raise Exception(msg)

        super().__init__(
            Comment("Logical SXdg"),
            qubit.SX(q),
        )


class SY(Block):
    """Square root of Y.

    X -> -Z
    Z -> X

    Y -> Y
    """

    def __init__(self, q: QReg) -> None:
        """Initialize a logical square root of Y gate on the Steane code.

        Args:
            q: A quantum register containing exactly 7 qubits representing a logical qubit
                in the Steane code.

        Raises:
            Exception: If the quantum register does not contain exactly 7 qubits.
        """
        if len(q.elems) != 7:
            msg = f"Size of register {len(q.elems)} != 7"
            raise Exception(msg)

        super().__init__(
            Comment("Logical SY"),
            qubit.SY(q),
        )


class SYdg(Block):
    """Square root of X.

    X -> Z
    Z -> -X

    Y -> Y
    """

    def __init__(self, q: QReg) -> None:
        """Initialize a logical adjoint square root of Y gate on the Steane code.

        Args:
            q: A quantum register containing exactly 7 qubits representing a logical qubit
                in the Steane code.

        Raises:
            Exception: If the quantum register does not contain exactly 7 qubits.
        """
        if len(q.elems) != 7:
            msg = f"Size of register {len(q.elems)} != 7"
            raise Exception(msg)

        super().__init__(
            Comment("Logical SYdg"),
            qubit.SYdg(q),
        )


class SZ(Block):
    """Square root of Z gate (S gate).

    Also known as the S gate with matrix representation diag(1, i).

    Action on Pauli operators:
    X -> Y
    Z -> Z
    Y -> -X
    """

    def __init__(self, q: QReg) -> None:
        """Initialize a logical square root of Z gate (S gate) on the Steane code.

        Args:
            q: A quantum register containing exactly 7 qubits representing a logical qubit
                in the Steane code.

        Raises:
            Exception: If the quantum register does not contain exactly 7 qubits.
        """
        if len(q.elems) != 7:
            msg = f"Size of register {len(q.elems)} != 7"
            raise Exception(msg)

        super().__init__(
            Comment("Logical SZ"),
            qubit.SZdg(q),
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
        """Initialize a logical adjoint square root of Z gate (S† gate) on the Steane code.

        Args:
            q: A quantum register containing exactly 7 qubits representing a logical qubit
                in the Steane code.

        Raises:
            Exception: If the quantum register does not contain exactly 7 qubits.
        """
        if len(q.elems) != 7:
            msg = f"Size of register {len(q.elems)} != 7"
            raise Exception(msg)

        super().__init__(
            Comment("Logical SZdg"),
            qubit.SZ(q),
        )
