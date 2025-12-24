"""T|+⟩ magic state preparation for the Steane 7-qubit code.

This module provides implementations for preparing the logical T|+⟩ magic state in the Steane 7-qubit code, which is
essential for implementing non-Clifford gates in fault-tolerant quantum computation.
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
from pecos.qeclib.steane.gates_sq.face_rots import F
from pecos.qeclib.steane.preps.encoding_circ import EncodingCircuit
from pecos.qeclib.steane.preps.plus_h_state import PrepHStateFT, PrepHStateFTRUS
from pecos.slr import Bit, Block, Comment, CReg, QReg
from pecos.slr.misc import Return
from pecos.slr.types import Array, QubitType


class PrepEncodeTPlusNonFT(Block):
    """Uses the encoding circuit to non-fault-tolerantly initialize the logical T|+> magic state.

    Returns:
        array[qubit, 7]: The encoded 7-qubit register in the T|+> state.
    """

    # Declare return type: returns the encoded qubit register
    block_returns = (Array[QubitType, 7],)

    def __init__(self, q: QReg) -> None:
        """Initialize PrepEncodeTPlusNonFT block for non-fault-tolerant T|+> preparation.

        Args:
            q: Quantum register containing 7 qubits for the Steane code.
        """
        super().__init__(
            Comment("Initialize logical |T> = T|+>\n============================="),
            qubit.Prep(q[6]),
            qubit.H(q[6]),
            qubit.T(q[6]),
            EncodingCircuit(q),
            # Explicitly declare return value
            Return(q),
        )


class PrepEncodeTDagPlusNonFT(Block):
    """Uses the encoding circuit to non-fault-tolerantly initialize the logical T†|+> magic state.

    Returns:
        array[qubit, 7]: The encoded 7-qubit register in the T†|+> state.
    """

    # Declare return type: returns the encoded qubit register
    block_returns = (Array[QubitType, 7],)

    def __init__(self, q: QReg) -> None:
        """Initialize PrepEncodeTDagPlusNonFT block for non-fault-tolerant T†|+> preparation.

        Args:
            q: Quantum register containing 7 qubits for the Steane code.
        """
        super().__init__(
            Comment("Initialize logical |T> = T|+>\n============================="),
            qubit.Prep(q[6]),
            qubit.H(q[6]),
            qubit.Tdg(q[6]),
            EncodingCircuit(q),
            # Explicitly declare return value
            Return(q),
        )


class PrepEncodeTPlusFT(Block):
    """Initialize a T|+> state fault tolerantly.

    Prepare |+H> by measuring the logical Hadamard, doing a QED round, and
    then rotate to T|+>.

    Returns:
        array[qubit, 7]: The encoded 7-qubit data register in the T|+> state.

    Arguments:
        d: Data qubits (size 7)
        a: Axillary qubits (size 2)
        out: Measurement outputs (size 2). out[0] is the Measure H result and out[1] is the flag result.
        reject: Whether the procedure failed and should be rejected. 0 it is good, 1 prep failed.
    """

    # Declare return type: returns the data qubit register
    block_returns = (Array[QubitType, 7],)

    def __init__(
        self,
        d: QReg,
        a: QReg,
        out: CReg,
        reject: Bit,
        flag_x: CReg,
        flag_z: CReg,
        flags: CReg,
        last_raw_syn_x: CReg,
        last_raw_syn_z: CReg,
    ) -> None:
        """Initialize PrepEncodeTPlusFT block for fault-tolerant T|+> preparation.

        Args:
            d: Data qubits (size 7) for the Steane code.
            a: Ancillary qubits (size 2) for measurements.
            out: Measurement outputs (size 2). out[0] is the Hadamard measurement,
                out[1] is the flag result.
            reject: Bit indicating preparation failure (0 for success, 1 for failure).
            flag_x: Classical register for X stabilizer flags.
            flag_z: Classical register for Z stabilizer flags.
            flags: Combined flags register.
            last_raw_syn_x: Previous X syndrome measurements.
            last_raw_syn_z: Previous Z syndrome measurements.
        """
        super().__init__(
            PrepHStateFT(
                d,
                a,
                out,
                reject,
                flag_x,
                flag_z,
                flags,
                last_raw_syn_x,
                last_raw_syn_z,
            ),
            F(d),  # |+H> -> T|+X>
            # Explicitly declare return value
            Return(d),
        )


class PrepEncodeTPlusFTRUS(Block):
    """Initialize a T|+> state fault tolerantly using repeat-until-success.

    By measuring the logical Hadamard using Repeat-until-success style
    initialization.

    Returns:
        array[qubit, 7]: The encoded 7-qubit data register in the T|+> state.

    Arguments:
        d: Data qubits (size 7)
        a: Axillary qubits (size 2)
        out: Measurement outputs (size 2). out[0] is the Measure H result and out[1] is the flag result.
        limit: The number of RUS steps to take.
        reject: Whether the procedure failed and should be rejected. 0 it is good, 1 prep failed.
    """

    # Declare return type: returns the data qubit register
    block_returns = (Array[QubitType, 7],)

    def __init__(
        self,
        d: QReg,
        a: QReg,
        out: CReg,
        reject: Bit,
        flag_x: CReg,
        flag_z: CReg,
        flags: CReg,
        last_raw_syn_x: CReg,
        last_raw_syn_z: CReg,
        limit: int,
    ) -> None:
        """Initialize PrepEncodeTPlusFTRUS block for repeat-until-success T|+> preparation.

        Args:
            d: Data qubits (size 7) for the Steane code.
            a: Ancillary qubits (size 2) for measurements.
            out: Measurement outputs (size 2). out[0] is the Hadamard measurement,
                out[1] is the flag result.
            reject: Bit indicating preparation failure (0 for success, 1 for failure).
            flag_x: Classical register for X stabilizer flags.
            flag_z: Classical register for Z stabilizer flags.
            flags: Combined flags register.
            last_raw_syn_x: Previous X syndrome measurements.
            last_raw_syn_z: Previous Z syndrome measurements.
            limit: Maximum number of preparation attempts.
        """
        # NOTE: For QASM, have to avoid nested If statements
        super().__init__(
            PrepHStateFTRUS(
                d,
                a,
                out,
                reject,
                flag_x,
                flag_z,
                flags,
                last_raw_syn_x,
                last_raw_syn_z,
                limit,
            ),
            F(d),
            # Explicitly declare return value
            Return(d),
        )
