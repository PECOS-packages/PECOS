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

from numpy import pi

from pecos.qeclib import qubit
from pecos.qeclib.generic.check_1flag import Check1Flag
from pecos.qeclib.steane.preps.encoding_circ import EncodingCircuit
from pecos.qeclib.steane.syn_extract.three_parallel_flagging import (
    ThreeParallelFlaggingXZZ,
    ThreeParallelFlaggingZXX,
)
from pecos.slr import Bit, Block, Comment, CReg, If, QReg, Repeat


class PrepHStateFT(Block):
    """
    Prepare a |+H> state fault tolerantly by using an encoding circuit to prepare logical|+H>, measuring the logical
    Hadamard with a flag, doing a QED round, and post-selecting based on non-trivial measurements.

    Arguments:
        d: Data qubits (size 7)
        a: Axillary qubits (size 2)
        out: Measurement outputs (size 2). out[0] is the Measure H result and out[1] is the flag result.
        reject: Whether the procedure failed and should be rejected. 0 it is good, 1 prep failed.
    """

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
        *,
        condition_qed: bool = True,
    ):
        super().__init__()

        # non-fault-tolerantly encode logical |+H>
        # ----------------------------------------
        self.extend(
            qubit.Prep(d[6]),
            qubit.RY[pi / 4](d[6]),
            EncodingCircuit(d),
        )

        # flagged logical H measurement
        # -----------------------------
        self.extend(
            Check1Flag(
                d=[d[0], d[1], d[2], d[3], d[4], d[5], d[6]],
                ops="HHHHHHH",
                a=a[0],
                flag=a[1],
                out=out[0],
                out_flag=out[1],
            ),
        )

        # QED
        self.extend(
            ThreeParallelFlaggingXZZ(
                d,
                a,
                flag_x,
                flag_z,
                flags,
                last_raw_syn_x,
                last_raw_syn_z,
            ),
        )

        if condition_qed:
            self.extend(
                If(flags == 0).Then(
                    ThreeParallelFlaggingZXX(
                        d,
                        a,
                        flag_x,
                        flag_z,
                        flags,
                        last_raw_syn_x,
                        last_raw_syn_z,
                    ),
                ),
            )
        else:
            self.extend(
                ThreeParallelFlaggingZXX(
                    d,
                    a,
                    flag_x,
                    flag_z,
                    flags,
                    last_raw_syn_x,
                    last_raw_syn_z,
                ),
            )

        self.extend(
            reject.set(out[0] | out[1] | flags[0] | flags[1] | flags[2]),
            # Reject on the results of the `reject` bit. 0 is good. 1 means the prep failed.
        )


class PrepHStateFTRUS(Block):
    """
    Prepare a |+H> state fault tolerantly by using an encoding circuit to prepare logical|+H>, measuring the logical
    Hadamard with a flag, doing a QED round, and post-selecting based on non-trivial measurements.

    Arguments:
        d: Data qubits (size 7)
        a: Axillary qubits (size 2)
        out: Measurement outputs (size 2). out[0] is the Measure H result and out[1] is the flag result.
        reject: Whether the procedure failed and should be rejected. 0 it is good, 1 prep failed.
    """

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
    ):
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
            Repeat(limit - 1).block(
                If(reject != 0).Then(
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
                        condition_qed=False,
                    ),
                ),
            ),
        )

        if limit == 1:
            self.extend(Comment())
