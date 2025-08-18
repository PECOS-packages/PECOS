"""Generic stabilizer check implementations.

This module provides generic implementations for stabilizer checks in
quantum error correction, including syndrome extraction circuits and
check operations that can be used across different QEC codes.
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

from __future__ import annotations

from typing import TYPE_CHECKING

from pecos.qeclib.qubit import CX, CY, CZ, H, Measure, Prep
from pecos.slr import Barrier, Block, Comment

if TYPE_CHECKING:
    from pecos.qeclib.qubit.qgate_base import QGate
    from pecos.slr import Bit, Qubit


class Check(Block):
    """Generic stabilizer check operation.

    This class implements a generic stabilizer check operation that applies
    a sequence of Pauli operators to data qubits controlled by an ancilla qubit.
    """

    def __init__(
        self,
        d: list[Qubit],
        paulis: str,
        a: Qubit,
        out: Bit,
        *,
        with_barriers: bool = False,
    ) -> None:
        """Initialize a stabilizer check measurement.

        Args:
            d: List of data qubits to check.
            paulis: String of Pauli operators (X, Y, or Z) to apply to each data qubit.
                Can be a single character (applied to all qubits) or one character per qubit.
            a: Ancilla qubit used for the check measurement.
            out: Classical bit to store the measurement result.
            with_barriers: Whether to insert barrier instructions between operations to prevent
                gate reordering. Defaults to False.

        Raises:
            Exception: If check weight is less than 2.
            Exception: If number of Paulis doesn't match number of data qubits.
            Exception: If invalid Pauli operator is specified.
        """
        super().__init__()

        n: int = len(d)

        if n <= 1:
            msg = "Check must be of weight 2 or more"
            raise Exception(msg)

        if len(paulis) == 1:
            paulis *= n
        elif n != len(paulis):
            msg = "Must use a single Pauli or have the same number of Paulis as data qubits."
            raise Exception(msg)

        for p in paulis:
            if p not in {"X", "Y", "Z"}:
                msg = 'Only "X", "Y" and "Z" are accepted.'
                raise Exception(msg)

        ps = paulis

        self.extend(
            Comment(f"Measure check {ps}"),
            Prep(a),
            H(a),
        )

        for i in range(n):
            if with_barriers:
                self.extend(
                    Barrier(a, d[i]),  # to preserve order
                )

            self.extend(
                self.cp(ps[i], a, d[i]),
            )

            if with_barriers:
                self.extend(
                    Barrier(a, d[i]),  # to preserve order
                )

        self.extend(
            H(a),
            Measure(a) > out,
        )

    @staticmethod
    def cp(p: str, a: Qubit, d: Qubit) -> QGate:
        """Create controlled Pauli gate based on string identifier.

        Args:
            p: Pauli gate identifier ('X', 'Y', or 'Z').
            a: Control qubit.
            d: Target qubit.

        Returns:
            Corresponding controlled Pauli gate.
        """
        if p == "X":
            return CX(a, d)
        if p == "Y":
            return CY(a, d)
        if p == "Z":
            return CZ(a, d)
        msg = f"Symbol '{p}' not supported!"
        raise Exception(msg)
