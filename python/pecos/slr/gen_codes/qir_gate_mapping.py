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
        ]
    )
    Fdg = QG.decompose(
        lambda fdg: [
            q.H(fdg.qargs[0]),
            q.SZ(fdg.qargs[0]),
        ]
    )
    F4 = QG.decompose(
        lambda f4: [
            q.H(f4.qargs[0]),
            q.SZdg(f4.qargs[0]),
        ]
    )
    F4dg = QG.decompose(
        lambda f4dg: [
            q.SZ(f4dg.qargs[0]),
            q.H(f4dg.qargs[0]),
        ]
    )

    # Remaining QIR gates seen:
    #     '__quantum__qis__rxxyyzz__body',
