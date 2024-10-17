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

from pecos.qeclib import qubit
from pecos.qeclib.steane.gates_sq.sqrt_paulis import SX, SYdg
from pecos.slr import Barrier, Block, Comment, If

if TYPE_CHECKING:
    from pecos.slr import Bit, CReg, QReg


class MeasureX(Block):
    """Measure in the logical X basis.

    Need: X_L -> Z_L

    SYdg: X->Z, Y->Y, Z->-X
    """

    def __init__(
        self,
        qubits: QReg,
        meas_creg: CReg,
        log_raw: Bit,
        *,
        barrier: bool = True,
    ):
        super().__init__()

        self.extend(
            SYdg(qubits),
            MeasureZ(qubits, meas_creg, log_raw, barrier=barrier),
        )


class MeasureY(Block):
    """Measure in the logical Y basis.

    Need: Y_L -> Z_L

    SX: X->X, Y->Z, Z->-Y
    """

    def __init__(
        self,
        qubits: QReg,
        meas_creg: CReg,
        log_raw: Bit,
        *,
        barrier: bool = True,
    ):
        super().__init__()

        self.extend(
            SX(qubits),  # logical SX == physical SXdg gates
            MeasureZ(qubits, meas_creg, log_raw, barrier=barrier),
        )


class MeasureZ(Block):
    """Measure in the logical Z basis."""

    def __init__(self, qubits: QReg, meas: CReg, log_raw: Bit, *, barrier: bool = True):
        super().__init__()

        q = qubits
        m = meas

        if barrier:
            self.extend(
                Comment(),
                Barrier(q),
            )

        self.extend(
            Comment(),
            qubit.Measure(q[0]) > m[0],
            qubit.Measure(q[1]) > m[1],
            qubit.Measure(q[2]) > m[2],
            qubit.Measure(q[3]) > m[3],
            qubit.Measure(q[4]) > m[4],
            qubit.Measure(q[5]) > m[5],
            qubit.Measure(q[6]) > m[6],
            Comment(),
            Comment("determine raw logical output"),
            Comment("============================"),
            log_raw.set(m[4] ^ m[5] ^ m[6]),
            Comment("\n"),
        )

        self.qubits = qubits
        self.meas = meas
        self.log_raw = log_raw
        self.barrier = barrier


class Measure(Block):
    def __init__(self, q: QReg, meas_creg: CReg, log_raw: Bit, meas_basis: str):
        super().__init__()

        if meas_basis == "X":
            self.extend(
                Comment("Destructive logical X measurement"),
                MeasureX(q, meas_creg, log_raw),
            )
        elif meas_basis == "Y":
            self.extend(
                Comment("Destructive logical Y measurement"),
                MeasureY(q, meas_creg, log_raw),
            )
        elif meas_basis == "Z":
            self.extend(
                Comment("Destructive logical Z measurement"),
                MeasureZ(q, meas_creg, log_raw),
            )
        else:
            msg = f"Logical measurement in '{meas_basis}' basis is not supported."
            raise Exception(msg)


class ProcessMeas(Block):
    """Process measurement results to determine additional corrections. Applies these and previous corrections to
    logical measurement."""

    def __init__(
        self,
        basis: str,
        meas: CReg,
        log_raw_bit: Bit,
        log_bit: Bit,
        syn_meas: CReg,
        pf_x: Bit,
        pf_z: Bit,
        check_type="xz",
        last_raw_syn_x: CReg | None = None,
        last_raw_syn_y: CReg | None = None,
        last_raw_syn_z: CReg | None = None,
        *,
        ft_meas: bool = True,
    ):
        super().__init__()

        log = log_bit
        log_raw = log_raw_bit

        self.extend(
            Comment(
                """
=================== //
PROCESS MEASUREMENT //
=================== //

Determine correction to get logical output
==========================================""",
            ),
            syn_meas[0].set(meas[0] ^ meas[1] ^ meas[2] ^ meas[3]),
            syn_meas[1].set(meas[1] ^ meas[2] ^ meas[4] ^ meas[5]),
            syn_meas[2].set(meas[2] ^ meas[3] ^ meas[5] ^ meas[6]),
            Comment("\nXOR syndromes"),
        )
        if check_type not in {"xy", "xz", "yz"}:
            msg = "QEC type not recognized!"
            raise Exception(msg)

        if basis == "X":
            if "x" in check_type:
                self.extend(
                    syn_meas.set(syn_meas ^ last_raw_syn_x),
                )
            else:  # yz
                self.extend(
                    syn_meas.set(syn_meas ^ last_raw_syn_y),
                    syn_meas.set(syn_meas ^ last_raw_syn_z),
                )
        elif basis == "Y":
            if "y" in check_type:
                self.extend(
                    syn_meas.set(syn_meas ^ last_raw_syn_y),
                )
            else:  # xz
                self.extend(
                    syn_meas.set(syn_meas ^ last_raw_syn_x),
                    syn_meas.set(syn_meas ^ last_raw_syn_z),
                )

        elif basis == "Z":
            if "z" in check_type:
                self.extend(
                    syn_meas.set(syn_meas ^ last_raw_syn_z),
                )
            else:  # xy
                self.extend(
                    syn_meas.set(syn_meas ^ last_raw_syn_x),
                    syn_meas.set(syn_meas ^ last_raw_syn_y),
                )

        else:
            msg = f"Measurement basis must be X, Y, or Z, not {basis}!"
            raise Exception(msg)

        if ft_meas:
            self.extend(
                Comment("\nCorrect logical output based on measured out syndromes"),
                log.set(log_raw),
                If(syn_meas == 2).Then(log.set(log ^ 1)),
                If(syn_meas == 4).Then(log.set(log ^ 1)),
                If(syn_meas == 6).Then(log.set(log ^ 1)),
                Comment(),
            )
        else:
            self.extend(
                Comment("\nnon-FT measure out"),
                log.set(log_raw),
            )

        self.extend(Comment("Apply Pauli frame update (flip the logical output)"))

        if basis == "X":
            self.extend(
                Comment("Update for logical X out"),
                log.set(log ^ pf_z),
            )
        elif basis == "Y":
            self.extend(
                Comment("Update for logical Y out"),
                log.set(log ^ pf_x),
                log.set(log ^ pf_z),
            )
        elif basis == "Z":
            self.extend(
                Comment("Update for logical Z out"),
                log.set(log ^ pf_x),
            )
        else:
            msg = f"Basis `{basis}` not supported!"
            raise Exception(msg)


def MeasDecode(
    q: QReg,
    meas_basis: str,
    meas: CReg,
    log_raw: Bit,
    log: Bit,
    syn_meas: CReg,
    pf_x: Bit,
    pf_z: Bit,
    last_raw_syn_x: CReg,
    last_raw_syn_z: CReg,
) -> Block:
    """Measure out in the appropriate logical basis, determine correction,
    apply to logical output."""
    return Block(
        Measure(q, meas, log_raw, meas_basis),
        ProcessMeas(
            meas_basis,
            meas,
            log_raw,
            log,
            syn_meas,
            pf_x,
            pf_z,
            last_raw_syn_z=last_raw_syn_z,
            last_raw_syn_x=last_raw_syn_x,
        ),
    )
