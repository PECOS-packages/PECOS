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

from enum import Enum

from pecos.qeclib import qubit as q
from pecos.qeclib.qubit.qgate_base import QGate
from dataclasses import dataclass, field

@dataclass
class QG:
    qir_name: str
    decomposition: list[QGate] = field(default_factory=list)
    adjoint: bool = False

    @classmethod
    def of(cls, *args):
        return cls('', args)

class QIRGateMetadata(Enum):
    """Maps QEClib gates to QIR gates, including possible decompositions. See:
        https://co41-bitbucket.honeywell.lab:4443/projects/HQSSW/repos/hqcompiler/browse/hqscompiler/qir/base_validator.py#72"""
    
    def __init__(self, gate: QG):
        self.qir_name = gate.qir_name
        self.decomposition = gate.decomposition

    H = QG('h')
    Y = QG('y')
    CX = QG('cnot')
    CZ = QG('cz')
    X = QG('x')
    Z = QG('z')
    RZ = QG('rz')
    RY = QG('ry')
    RX = QG('rx')
    S = QG('s')
    T = QG('t')
    R1XY = QG('u1q')
    RZZ = QG('rzz')
    SZZ = QG('zz')

    # These dagger/adjoint gates are generated with slightly different names
    Sdg = QG('s__adj')
    Tdg = QG('t__adj')

    SZ = S # Mapped to itself in qeclib
    SZdg = Sdg # Mapped to itself in qeclib
    
    F = QG.of(q.SZdg(), q.H())
    Fdg = QG.of(q.H(), q.SZ()) 
    F4 = QG.of(q.H(), q.SZdg())
    F4dg = QG.of(q.SZ(), q.H())

    # Remaining QIR gates:

    #     '__quantum__qis__rxxyyzz__body',
    #     # for tket support
    #     '__quantum__qis__zzmax__body',
    #     '__quantum__qis__phasedx__body',
    #     '__quantum__qis__zzphase__body',
    #     '__quantum__qis__tk2__body',