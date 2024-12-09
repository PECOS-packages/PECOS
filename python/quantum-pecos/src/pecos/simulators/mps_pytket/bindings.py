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

import pecos.simulators.mps_pytket.gates_one_qubit as one_q
import pecos.simulators.mps_pytket.gates_two_qubit as two_q
from pecos.simulators.mps_pytket.gates_init import init_one, init_zero
from pecos.simulators.mps_pytket.gates_meas import meas_z

# Supporting gates from table:
#   https://github.com/CQCL/phir/blob/main/spec.md#table-ii---quantum-operations

gate_dict = {
    "Init": init_zero,
    "Init +Z": init_zero,
    "Init -Z": init_one,
    "init |0>": init_zero,
    "init |1>": init_one,
    "leak": init_zero,
    "leak |0>": init_zero,
    "leak |1>": init_one,
    "unleak |0>": init_zero,
    "unleak |1>": init_one,
    "Measure": meas_z,
    "measure Z": meas_z,
    "I": one_q.identity,
    "X": one_q.X,
    "Y": one_q.Y,
    "Z": one_q.Z,
    "RX": one_q.RX,
    "RY": one_q.RY,
    "RZ": one_q.RZ,
    "R1XY": one_q.R1XY,
    "RXY1Q": one_q.R1XY,
    "SX": one_q.SX,
    "SXdg": one_q.SXdg,
    "SqrtX": one_q.SX,
    "SqrtXd": one_q.SXdg,
    "SY": one_q.SY,
    "SYdg": one_q.SYdg,
    "SqrtY": one_q.SY,
    "SqrtYd": one_q.SYdg,
    "SZ": one_q.SZ,
    "SZdg": one_q.SZdg,
    "SqrtZ": one_q.SZ,
    "SqrtZd": one_q.SZdg,
    "H": one_q.H,
    "F": one_q.F,
    "Fdg": one_q.Fdg,
    "T": one_q.T,
    "Tdg": one_q.Tdg,
    "CX": two_q.CX,
    "CY": two_q.CY,
    "CZ": two_q.CZ,
    "RXX": two_q.RXX,
    "RYY": two_q.RYY,
    "RZZ": two_q.RZZ,
    "R2XXYYZZ": two_q.R2XXYYZZ,
    "SXX": two_q.SXX,
    "SXXdg": two_q.SXXdg,
    "SYY": two_q.SYY,
    "SYYdg": two_q.SYYdg,
    "SZZ": two_q.SZZ,
    "SqrtZZ": two_q.SZZ,
    "SZZdg": two_q.SZZdg,
    "SWAP": two_q.SWAP,
    # Additional Cliffords from `circuit_converters/std2chs.py`
    "Q": one_q.SX,
    "Qd": one_q.SXdg,
    "R": one_q.SY,
    "Rd": one_q.SYdg,
    "S": one_q.SZ,
    "Sd": one_q.SZdg,
    "H1": one_q.H,
    "H2": one_q.H2,
    "H3": one_q.H3,
    "H4": one_q.H4,
    "H5": one_q.H5,
    "H6": one_q.H6,
    "H+z+x": one_q.H,
    "H-z-x": one_q.H2,
    "H+y-z": one_q.H3,
    "H-y-z": one_q.H4,
    "H-x+y": one_q.H5,
    "H-x-y": one_q.H6,
    "F1": one_q.F,
    "F1d": one_q.Fdg,
    "F2": one_q.F2,
    "F2d": one_q.F2d,
    "F3": one_q.F3,
    "F3d": one_q.F3d,
    "F4": one_q.F4,
    "F4d": one_q.F4d,
    "CNOT": two_q.CX,
    "G": two_q.G,
    "II": one_q.identity,
}
