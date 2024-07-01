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
from pecos.qeclib.steane.gates_sq.hadarmards import H
from pecos.qeclib.steane.gates_sq.paulis import X, Z
from pecos.slr import QASM, Barrier, Bit, Block, If, QReg, Qubit, Repeat, util


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
            QASM(""),
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


class PrepZeroVerify:
    """Verify the initialization of InitEncodingNonFTZero"""

    def __init__(self, qubits: QReg, ancilla: Qubit, init_bit: Bit, *, reset_ancilla: bool = True):
        self.qubits = qubits
        self.ancilla = ancilla
        self.init_bit = init_bit
        self.reset_ancilla = reset_ancilla

    def qasm(self):
        q = self.qubits
        a = self.ancilla
        c = self.init_bit

        qasm = f"""
        barrier {a},{q[1]},{q[3]},{q[5]};
        //verification step
        """

        if self.reset_ancilla:
            qasm += f"\nreset {a};\n"

        qasm += f"""cx {q[5]},{a};
        cx {q[1]},{a};
        cx {q[3]},{a};
        measure {a} -> {c};
        """

        return util.rm_white_space(qasm)


class PrepEncodingFTZero(Block):
    """Represents the non-fault-tolerant encoding circuit for the Steane code.

    Args:
        data (QReg[7]):
        ancilla (Qubit):
        init_bit (Bit):
        reset (bool):
    """

    def __init__(self, data: QReg, ancilla: Qubit, init_bit: Bit, *, reset: bool = True):
        q = data
        a = ancilla

        super().__init__()

        self.extend(
            QASM(""),
            Barrier(q[0], q[1], q[2], q[3], q[4], q[5], q[6], a),
            QASM(""),
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
            # Rotate to the Paulli basis of choice
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
            case "|0>" | "+Z":
                pass
            case "|+>" | "+X":
                self.extend(
                    H(q),
                )
            case "|->" | "-X":
                self.extend(
                    H(q),
                    Z(q),
                )
            case "|+i>" | "+Y":
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
