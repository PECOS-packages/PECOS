"""Transversal two-qubit logical gates for the Steane 7-qubit code.

This module provides transversal two-qubit logical gate implementations for the Steane 7-qubit code, enabling
fault-tolerant operations between logical qubits while preserving error correction properties.
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

from pecos.slr import Barrier, Block, Comment, QReg
from pecos.slr.qeclib import qubit


class CX(Block):
    """Transversal logical CX gate for Steane code.

    This class implements a transversal logical controlled-X gate between
    two logical qubits encoded in the Steane code.
    """

    # Declare resource effects so SLR -> AST emits BlockDecl + BlockCall
    # instead of inlining. Both registers are live_preserved -- the body applies a
    # transversal CX pairwise (no internal measurements).
    block_inputs: ClassVar[dict[str, str]] = {
        "q1": "live_preserved",
        "q2": "live_preserved",
    }

    def __init__(self, q1: QReg, q2: QReg, *, barrier: bool = True) -> None:
        """Initialize a transversal logical CX gate on two Steane code logical qubits.

        Args:
            q1: First quantum register containing exactly 7 qubits (control).
            q2: Second quantum register containing exactly 7 qubits (target).
            barrier: Whether to include barriers before and after the gate operation.
                Defaults to True.

        Raises:
            Exception: If either quantum register does not contain exactly 7 qubits.
        """
        if len(q1.elems) != 7:
            msg = f"Size of register {len(q1.elems)} != 7"
            raise Exception(msg)

        if len(q2.elems) != 7:
            msg = f"Size of register {len(q2.elems)} != 7"
            raise Exception(msg)

        super().__init__()
        # SLR -> AST converter reads these to bind block_inputs param names to outer-scope refs.
        self.q1 = q1
        self.q2 = q2
        self.extend(
            Comment("Transversal Logical CX"),
        )
        if barrier:
            self.extend(
                Barrier(q1, q2),
            )
        self.extend(
            qubit.CX(
                (q1[0], q2[0]),
                (q1[1], q2[1]),
                (q1[2], q2[2]),
                (q1[3], q2[3]),
                (q1[4], q2[4]),
                (q1[5], q2[5]),
                (q1[6], q2[6]),
            ),
        )
        if barrier:
            self.extend(
                Barrier(q1, q2),
            )


class CY(Block):
    """Transversal logical CY gate for Steane code.

    This class implements a transversal logical controlled-Y gate between
    two logical qubits encoded in the Steane code.
    """

    block_inputs: ClassVar[dict[str, str]] = {
        "q1": "live_preserved",
        "q2": "live_preserved",
    }

    def __init__(self, q1: QReg, q2: QReg) -> None:
        """Initialize a transversal logical CY gate on two Steane code logical qubits.

        Args:
            q1: First quantum register containing exactly 7 qubits (control).
            q2: Second quantum register containing exactly 7 qubits (target).

        Raises:
            Exception: If either quantum register does not contain exactly 7 qubits.
        """
        if len(q1.elems) != 7:
            msg = f"Size of register {len(q1.elems)} != 7"
            raise Exception(msg)

        if len(q2.elems) != 7:
            msg = f"Size of register {len(q2.elems)} != 7"
            raise Exception(msg)

        super().__init__(
            Comment("Transversal Logical CY"),
            Barrier(q1, q2),
            qubit.CY(
                (q1[0], q2[0]),
                (q1[1], q2[1]),
                (q1[2], q2[2]),
                (q1[3], q2[3]),
                (q1[4], q2[4]),
                (q1[5], q2[5]),
                (q1[6], q2[6]),
            ),
            Barrier(q1, q2),
        )
        self.q1 = q1
        self.q2 = q2


class CZ(Block):
    """Transversal logical CZ gate for Steane code.

    This class implements a transversal logical controlled-Z gate between
    two logical qubits encoded in the Steane code.
    """

    block_inputs: ClassVar[dict[str, str]] = {
        "q1": "live_preserved",
        "q2": "live_preserved",
    }

    def __init__(self, q1: QReg, q2: QReg) -> None:
        """Initialize a transversal logical CZ gate on two Steane code logical qubits.

        Args:
            q1: First quantum register containing exactly 7 qubits.
            q2: Second quantum register containing exactly 7 qubits.

        Raises:
            Exception: If either quantum register does not contain exactly 7 qubits.
        """
        if len(q1.elems) != 7:
            msg = f"Size of register {len(q1.elems)} != 7"
            raise Exception(msg)

        if len(q2.elems) != 7:
            msg = f"Size of register {len(q2.elems)} != 7"
            raise Exception(msg)

        super().__init__(
            Comment("Transversal Logical CZ"),
            Barrier(q1, q2),
            qubit.CZ(
                (q1[0], q2[0]),
                (q1[1], q2[1]),
                (q1[2], q2[2]),
                (q1[3], q2[3]),
                (q1[4], q2[4]),
                (q1[5], q2[5]),
                (q1[6], q2[6]),
            ),
            Barrier(q1, q2),
        )
        self.q1 = q1
        self.q2 = q2


class SZZ(Block):
    """Transversal logical SZZ gate for Steane code.

    This class implements a transversal logical SZZ interaction gate between
    two logical qubits encoded in the Steane code.
    """

    def __init__(self, q1: QReg, q2: QReg) -> None:
        """Initialize a transversal logical SZZ gate on two Steane code logical qubits.

        Args:
            q1: First quantum register containing exactly 7 qubits.
            q2: Second quantum register containing exactly 7 qubits.

        Raises:
            Exception: If either quantum register does not contain exactly 7 qubits.
        """
        if len(q1.elems) != 7:
            msg = f"Size of register {len(q1.elems)} != 7"
            raise Exception(msg)

        if len(q2.elems) != 7:
            msg = f"Size of register {len(q2.elems)} != 7"
            raise Exception(msg)

        super().__init__(
            Comment("Transversal Logical SZZ"),
            Barrier(q1, q2),
            qubit.SZZ(
                (q1[0], q2[0]),
                (q1[1], q2[1]),
                (q1[2], q2[2]),
                (q1[3], q2[3]),
                (q1[4], q2[4]),
                (q1[5], q2[5]),
                (q1[6], q2[6]),
            ),
            Barrier(q1, q2),
        )


class SZZdg(Block):
    """Transversal logical SZZdg gate for Steane code.

    This class implements a transversal logical SZZ interaction gate between
    two logical qubits encoded in the Steane code.
    """

    def __init__(self, q1: QReg, q2: QReg) -> None:
        """Initialize a transversal logical SZZdg gate on two Steane code logical qubits.

        Args:
            q1: First quantum register containing exactly 7 qubits.
            q2: Second quantum register containing exactly 7 qubits.

        Raises:
            Exception: If either quantum register does not contain exactly 7 qubits.
        """
        if len(q1.elems) != 7:
            msg = f"Size of register {len(q1.elems)} != 7"
            raise Exception(msg)

        if len(q2.elems) != 7:
            msg = f"Size of register {len(q2.elems)} != 7"
            raise Exception(msg)

        super().__init__(
            Comment("Transversal Logical SZZdg"),
            Barrier(q1, q2),
            qubit.SZZdg(
                (q1[0], q2[0]),
                (q1[1], q2[1]),
                (q1[2], q2[2]),
                (q1[3], q2[3]),
                (q1[4], q2[4]),
                (q1[5], q2[5]),
                (q1[6], q2[6]),
            ),
            Barrier(q1, q2),
        )
