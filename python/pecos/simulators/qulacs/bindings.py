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

import pecos.simulators.qulacs.gates_one_qubit as one_q
import pecos.simulators.qulacs.gates_two_qubit as two_q
from pecos.simulators.qulacs.gates_init import init_one, init_zero
from pecos.simulators.qulacs.gates_meas import meas_z

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
}
