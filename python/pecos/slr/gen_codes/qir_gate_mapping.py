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

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from numpy import pi

from pecos.qeclib import qubit as q
from pecos.qeclib.qubit.qgate_base import QGate


@dataclass
class QG:
    qir_name: str
    adjoint: bool = False
    decomposer: Callable[[QGate], list[QGate]] = None

    @classmethod
    def decompose(cls, decomposer: Callable[[QGate], list[QGate]]):
        return cls("", False, decomposer)


class QIRGateMetadata(Enum):
    """Maps QEClib gates to QIR gates, including possible decompositions. See:
    https://co41-bitbucket.honeywell.lab:4443/projects/HQSSW/repos/hqcompiler/browse/hqscompiler/qir/base_validator.py#72
    """

    def __init__(self, gate: QG):
        self.qir_name = gate.qir_name
        self.decomposer = gate.decomposer

    H = QG("h")
    Y = QG("y")
    CX = QG("cnot")
    CZ = QG("cz")
    X = QG("x")
    Z = QG("z")
    RZ = QG("rz")
    RY = QG("ry")
    RX = QG("rx")
    S = QG("s")
    T = QG("t")
    R1XY = QG("u1q")
    RZZ = QG("rzz")
    SZZ = QG("zz")

    # These dagger/adjoint gates are generated with slightly different names
    Sdg = QG("s__adj")
    Tdg = QG("t__adj")

    SZ = S  # Mapped to itself in qeclib
    SZdg = Sdg  # Mapped to itself in qeclib

    # Complicated gates that require decomposition, expressed as a lambda:
    F = QG.decompose(
        lambda f: [
            q.SZdg(f.qargs[0]),
            q.H(f.qargs[0]),
        ],
    )
    Fdg = QG.decompose(
        lambda fdg: [
            q.H(fdg.qargs[0]),
            q.SZ(fdg.qargs[0]),
        ],
    )
    F4 = QG.decompose(
        lambda f4: [
            q.H(f4.qargs[0]),
            q.SZdg(f4.qargs[0]),
        ],
    )
    F4dg = QG.decompose(
        lambda f4dg: [
            q.SZ(f4dg.qargs[0]),
            q.H(f4dg.qargs[0]),
        ],
    )

    # CH q1, q2; = RY(pi/4) q2; CZ q1, q2; RY(-pi/4) q2;
    CH = QG.decompose(
        lambda ch: [
            q.RY[pi / 4](ch.qargs[1]),
            q.CZ(ch.qargs[0], ch.qargs[1]),
            q.RY[-pi / 4](ch.qargs[1]),
        ],
    )

    # CY q1,q2 = S q2; CX q1,q2; S_adj q2;
    CY = QG.decompose(
        lambda cy: [
            q.S(cy.qargs[1]),
            q.CX(cy.qargs[0], cy.qargs[1]),
            q.Sdg(cy.qargs[1]),
        ],
    )

    # SXX q1, q2 = RX(pi/2) q1; RX(pi/2) q2; RY(-pi/2) q1; CX q1, q2; RY(pi/2) q1;
    SXX = QG.decompose(
        lambda sxx: [
            q.RX[pi / 2](sxx.qargs[0]),
            q.RX[pi / 2](sxx.qargs[1]),
            q.RY[-pi / 2](sxx.qargs[0]),
            q.CX(sxx.qargs[0], sxx.qargs[1]),
            q.RY[pi / 2](sxx.qargs[0]),
        ],
    )

    # SYY q1, q2 = S_adj q1; S_adj q2; SXX q1, q2; S q1; S q2;
    SYY = QG.decompose(
        lambda syy: [
            q.Sdg(syy.qargs[0]),
            q.Sdg(syy.qargs[1]),
            q.SXX(syy.qargs[0], syy.qargs[1]),
            q.S(syy.qargs[0]),
            q.S(syy.qargs[1]),
        ],
    )

    # SXXdg q1, q2 = X q1; X q2; SXX q1, q2;
    SXXdg = QG.decompose(
        lambda sxxdg: [
            q.X(sxxdg.qargs[0]),
            q.X(sxxdg.qargs[1]),
            q.SXX(sxxdg.qargs[0], sxxdg.qargs[1]),
        ],
    )

    # SYYdg q1, q2 = Y q1; Y q2; SYY q1, q2;
    SYYdg = QG.decompose(
        lambda syydg: [
            q.Y(syydg.qargs[0]),
            q.Y(syydg.qargs[1]),
            q.SYY(syydg.qargs[0], syydg.qargs[1]),
        ],
    )

    # SZZdg = Z q1; Z q2; ZZ (ZZMax?) q1, q2;
    SZZdg = QG.decompose(
        lambda szzdg: [
            q.Z(szzdg.qargs[0]),
            q.Z(szzdg.qargs[1]),
            q.SZZ(szzdg.qargs[0], szzdg.qargs[1]),
        ],
    )

    # SXdg = RX(-pi/2)
    SXdg = QG.decompose(
        lambda sxxdg: [
            q.RX[-pi / 2](sxxdg.qargs[0]),
        ],
    )

    # SYdg = RY(-pi/2)
    SYdg = QG.decompose(
        lambda syydg: [
            q.RY[-pi / 2](syydg.qargs[0]),
        ],
    )

    # SX = RX(pi/2)
    SX = QG.decompose(
        lambda sx: [
            q.RX[pi / 2](sx.qargs[0]),
        ],
    )
