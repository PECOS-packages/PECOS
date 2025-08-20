"""Single-flag stabilizer check implementations.

This module provides stabilizer check implementations with single flag
qubits for fault-tolerant syndrome extraction, enabling detection of
errors that occur during the measurement process.
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

from pecos.qeclib.qubit import CH, CX, CY, CZ, H, Measure, Prep
from pecos.slr import Barrier, Block, Comment

if TYPE_CHECKING:
    from pecos.qeclib.qubit.qgate_base import QGate
    from pecos.slr import Bit, Qubit


class Check1Flag(Block):
    """Single-flag stabilizer check operation.

    This class implements a stabilizer check operation with a single flag qubit
    to detect errors during the syndrome extraction process.
    """

    def __init__(
        self,
        d: list[Qubit],
        ops: str,
        a: Qubit,
        flag: Qubit,
        out: Bit,
        out_flag: Bit,
        *,
        with_barriers: bool = False,
    ) -> None:
        """Initialize a stabilizer check measurement with flag qubit.

        Args:
            d: List of data qubits to check.
            ops: String of operators (X, Y, Z, or H) to apply to each data qubit.
                Can be a single character (applied to all qubits) or one character per qubit.
            a: Ancilla qubit used for the check measurement.
            flag: Flag qubit used to detect hook errors.
            out: Classical bit to store the measurement result.
            out_flag: Classical bit to store the flag measurement result.
            with_barriers: Whether to insert barrier instructions between operations to prevent
                gate reordering. Defaults to False.

        Raises:
            Exception: If check weight is less than 3.
            Exception: If number of operators doesn't match number of data qubits.
            Exception: If invalid operator is specified.
        """
        super().__init__()

        n: int = len(d)

        if n <= 2:
            msg = "Check must be of weight 3 or more"
            raise Exception(msg)

        if len(ops) == 1:
            ops *= n
        elif n != len(ops):
            msg = "Must use a single operator or have the same number of operators as data qubits."
            raise Exception(msg)

        for o in ops:
            if o not in {"X", "Y", "Z", "H"}:
                msg = 'Only "X", "Y", "Z", and "H" are accepted.'
                raise Exception(msg)

        self.extend(
            Comment(f"Measure check {ops}"),
            Prep(a, flag),
            H(a),
        )
        if with_barriers:
            self.extend(
                Barrier(a, d[0]),
            )
        self.extend(
            self.cu(ops[0], a, d[0]),
        )
        if with_barriers:
            self.extend(
                Barrier(a, flag),
            )
        self.extend(
            CX(a, flag),
        )
        if with_barriers:
            self.extend(
                Barrier(a, flag),
            )

        for i in range(1, n - 1):
            self.extend(
                self.cu(ops[i], a, d[i]),
            )
            if with_barriers:
                self.extend(
                    Barrier(a, d[i]),  # To preserve order
                )

        if with_barriers:
            self.extend(
                Barrier(a, flag),
            )
        self.extend(
            CX(a, flag),
        )
        if with_barriers:
            self.extend(
                Barrier(a, flag),
            )
        self.extend(
            self.cu(ops[-1], a, d[-1]),
        )
        if with_barriers:
            self.extend(
                Barrier(a, d[-1]),
            )
        self.extend(
            H(a),
            Measure(a) > out,
            Measure(flag) > out_flag,
        )

    @staticmethod
    def cu(u: str, a: Qubit, d: Qubit) -> QGate:
        """Create controlled unitary gate based on string identifier.

        Args:
            u: Unitary gate identifier ('X', 'Y', 'Z', or 'H').
            a: Control qubit.
            d: Target qubit.

        Returns:
            Corresponding controlled unitary gate.
        """
        if u == "X":
            return CX(a, d)
        if u == "Y":
            return CY(a, d)
        if u == "Z":
            return CZ(a, d)
        if u == "H":
            return CH(a, d)
        msg = f"Symbol '{u}' not supported!"
        raise Exception(msg)
