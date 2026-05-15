"""Logical Pauli gates for the Steane 7-qubit code.

This module provides logical Pauli gate implementations (X, Y, Z) for the Steane 7-qubit code, implemented as
transversal operations that preserve the quantum error correction properties.
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

from typing import ClassVar

from pecos.slr import Block, Comment, QReg
from pecos.slr.qeclib import qubit


class X(Block):
    """Pauli X.

    X -> X
    Z -> -Z

    Y -> -Y
    """

    block_inputs: ClassVar[dict[str, str]] = {"q": "live_preserved"}

    def __init__(self, q: QReg) -> None:
        """Initialize a logical Pauli X gate on the Steane code.

        Args:
            q: A quantum register containing exactly 7 qubits representing a logical qubit
                in the Steane code.

        Raises:
            Exception: If the quantum register does not contain exactly 7 qubits.
        """
        if len(q.elems) != 7:
            msg = f"Size of register {len(q.elems)} != 7"
            raise Exception(msg)

        super().__init__()
        self.q = q
        self.extend(
            Comment("Logical X"),
            qubit.X(q[4]),
            qubit.X(q[5]),
            qubit.X(q[6]),
        )


class Y(Block):
    """Pauli Y.

    X -> -X
    Z -> -Z

    Y -> Y
    """

    block_inputs: ClassVar[dict[str, str]] = {"q": "live_preserved"}

    def __init__(self, q: QReg) -> None:
        """Initialize a logical Pauli Y gate on the Steane code.

        Args:
            q: A quantum register containing exactly 7 qubits representing a logical qubit
                in the Steane code.

        Raises:
            Exception: If the quantum register does not contain exactly 7 qubits.
        """
        if len(q.elems) != 7:
            msg = f"Size of register {len(q.elems)} != 7"
            raise Exception(msg)

        super().__init__()
        self.q = q
        self.extend(
            Comment("Logical Y"),
            qubit.Y(q[4]),
            qubit.Y(q[5]),
            qubit.Y(q[6]),
        )


class Z(Block):
    """Pauli Z.

    X -> -X
    Z -> Z

    Y -> -Y
    """

    block_inputs: ClassVar[dict[str, str]] = {"q": "live_preserved"}

    def __init__(self, q: QReg) -> None:
        """Initialize a logical Pauli Z gate on the Steane code.

        Args:
            q: A quantum register containing exactly 7 qubits representing a logical qubit
                in the Steane code.

        Raises:
            Exception: If the quantum register does not contain exactly 7 qubits.
        """
        if len(q.elems) != 7:
            msg = f"Size of register {len(q.elems)} != 7"
            raise Exception(msg)

        super().__init__()
        self.q = q
        self.extend(
            Comment("Logical Z"),
            qubit.Z(q[4]),
            qubit.Z(q[5]),
            qubit.Z(q[6]),
        )
