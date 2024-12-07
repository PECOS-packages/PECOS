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
from pecos.qeclib.qubit import Prep
from pecos.qeclib.steane.gates_sq import sqrt_paulis
from pecos.qeclib.steane.gates_sq.hadamards import H
from pecos.qeclib.steane.gates_sq.paulis import X, Z
from pecos.slr import Barrier, Bit, Block, Comment, If, QReg, Qubit, Repeat


class PrepEncodingNonFTZero(Block):
    """Represents the non-fault-tolerant encoding circuit for the Steane code."""

    def __init__(self, q: QReg):
        if len(q.elems) != 7:
            msg = f"Size of register {len(q.elems)} != 7"
            raise Exception(msg)

        super().__init__(
            qubit.H(
                q[0],
                q[4],
                q[6],
            ),
            Comment(),
            qubit.CX(
                (q[4], q[5]),
                (q[0], q[1]),
                (q[6], q[3]),
                (q[4], q[2]),
                (q[6], q[5]),
                (q[0], q[3]),
                (q[4], q[1]),
                (q[3], q[2]),
            ),
        )


class PrepZeroVerify(Block):
    """Verify the initialization of InitEncodingNonFTZero"""

    def __init__(
        self,
        qubits: QReg,
        ancilla: Qubit,
        init_bit: Bit,
        *,
        reset_ancilla: bool = True,
    ):
        q = qubits
        a = ancilla
        c = init_bit

        super().__init__(
            Comment(),
            Barrier(a, q[1], q[3], q[5]),
            Comment("verification step"),
        )

        if reset_ancilla:
            self.extend(
                Comment(),
                Prep(a),
            )

        self.extend(
            qubit.CX(
                (q[5], a),
                (q[1], a),
                (q[3], a),
            ),
            qubit.Measure(a) > c,
            Comment(""),
        )


class PrepEncodingFTZero(Block):
    """Represents the non-fault-tolerant encoding circuit for the Steane code.

    Args:
        data (QReg[7]):
        ancilla (Qubit):
        init_bit (Bit):
        reset (bool):
    """

    def __init__(
        self,
        data: QReg,
        ancilla: Qubit,
        init_bit: Bit,
        *,
        reset: bool = True,
    ):
        q = data
        a = ancilla

        super().__init__()

        self.extend(
            Comment(),
            Barrier(q[0], q[1], q[2], q[3], q[4], q[5], q[6], a),
            Comment(),
        )

        if reset:
            self.extend(
                Prep(q),
                Prep(a),
                Barrier(q, a),
            )

        self.extend(
            PrepEncodingNonFTZero(data),
            # reset_ancilla to False because it is reset earlier
            PrepZeroVerify(data, ancilla, init_bit, reset_ancilla=False),
        )


class PrepRUS(Block):
    """Use repeat-until-success to initialize a logical qubit."""

    def __init__(
        self,
        q: QReg,
        a: Qubit,
        init: Bit,
        limit: int,
        state: str = "|0>",
        *,
        first_round_reset: bool = True,
    ):
        super().__init__(
            PrepEncodingFTZero(q, a, init, reset=first_round_reset),
            Repeat(limit - 1).block(
                If(init == 1).Then(
                    PrepEncodingFTZero(q, a, init, reset=True),
                ),
            ),
        )
        if limit == 1:
            self.extend(
                Comment(),
            )

        self.extend(
            # Rotate to the Pauli basis of choice
            LogZeroRot(q, state),
        )


class LogZeroRot(Block):
    """Rotate logical |0> to appropriate Pauli state."""

    def __init__(self, q: QReg, state: str):
        super().__init__()

        match state:
            case "|1>" | "-Z":
                self.extend(
                    X(q),
                )
            case "|0>" | "+Z" | "Z":
                pass
            case "|+>" | "+X" | "X":
                self.extend(
                    H(q),
                )
            case "|->" | "-X":
                self.extend(
                    H(q),
                    Z(q),
                )
            case "|+i>" | "+Y" | "Y":
                self.extend(
                    sqrt_paulis.SXdg(q),
                )
            case "|-i>" | "-Y":
                self.extend(
                    sqrt_paulis.SX(q),
                )
            case _:
                msg = f"Unsupported init state '{state}'"
                raise Exception(msg)
