"""Encoding circuit implementations for the Steane 7-qubit code.

This module provides encoding circuit implementations that transform single logical qubits into the 7-qubit Steane code
representation, enabling quantum error correction.
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

from pecos.qeclib import qubit
from pecos.slr import Block, Comment, QReg
from pecos.slr.misc import Return
from pecos.slr.types import Array, QubitType


class EncodingCircuit(Block):
    """Encoding circuit for Steane code.

    This class implements the encoding circuit that transforms a single logical
    qubit into the 7-qubit Steane code representation.

    Returns:
        array[qubit, 7]: The encoded 7-qubit register.
    """

    # Declare return type: returns the encoded qubit register
    block_returns = (Array[QubitType, 7],)

    def __init__(self, q: QReg) -> None:
        """Initialize EncodingCircuit block for Steane code encoding.

        Args:
            q: Quantum register containing 7 qubits for the Steane code.
                The qubit at index 6 is the input qubit to be encoded.
        """
        self.q = q
        super().__init__()
        self.extend(
            Comment("\nEncoding circuit"),
            Comment("---------------"),
            qubit.Prep(
                q[0],
                q[1],
                q[2],
                q[3],
                q[4],
                q[5],
            ),
            Comment("\nq[6] is the input qubit\n"),
            qubit.CX(q[6], q[5]),
            Comment(""),
            qubit.H(q[1]),
            qubit.CX(q[1], q[0]),
            Comment(""),
            qubit.H(q[2]),
            qubit.CX(q[2], q[4]),
            Comment("\n---------------"),
            qubit.H(q[3]),
            qubit.CX(
                (q[3], q[5]),
                (q[2], q[0]),
                (q[6], q[4]),
            ),
            Comment("\n---------------"),
            qubit.CX(
                (q[2], q[6]),
                (q[3], q[4]),
                (q[1], q[5]),
            ),
            Comment("\n---------------"),
            qubit.CX(
                (q[1], q[6]),
                (q[3], q[0]),
            ),
            Comment(""),
            # Explicitly declare return value (like Python's return statement)
            # Combined with block_returns annotation for robust type checking
            Return(q),
        )
