# Copyright 2022 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from . import gates_sq
from . import gates_tq
from . import gates_meas
from . import gates_init

gate_dict = {

    'U1q': gates_sq.U1q,
    'RXY1Q': gates_sq.U1q,

    'RX': gates_sq.RX,
    'RY': gates_sq.RY,
    'RZ': gates_sq.RZ,

    'H': gates_sq.H,

    'I': gates_sq.I,
    'X': gates_sq.X,
    'Y': gates_sq.Y,
    'Z': gates_sq.Z,

    'SqrtZZ': gates_tq.SqrtZZ,
    'RZZ': gates_tq.RZZ,
    'CNOT': gates_tq.CX,
    'CX': gates_tq.CX,

    'measure Z': gates_meas.Measure,

    'init |0>': gates_init.init_zero,  # Init by measuring (if entangle => random outcome
    'init |1>': gates_init.init_one,  # Init by measuring (if entangle => random outcome

    'leak': gates_init.init_zero,
    'unleak |0>': gates_init.init_zero,
    'unleak |1>': gates_init.init_one,

    # Square root of Paulis
    'Q': gates_sq.Q,    # +x-y  sqrt of X
    'Qd': gates_sq.Qd,  # +x+y sqrt of X dagger
    'R': gates_sq.R,    # -z+x sqrt of Y
    'Rd': gates_sq.Rd,  # +z-x sqrt of Y dagger
    'S': gates_sq.S,    # +y+z sqrt of Z
    'Sd': gates_sq.Sd,  # -y+z sqrt of Z dagger

    'SqrtX': gates_sq.Q,    # +x-y  sqrt of X
    'SqrtXd': gates_sq.Qd,  # +x+y sqrt of X dagger
    'SqrtY': gates_sq.R,    # -z+x sqrt of Y
    'SqrtYd': gates_sq.Rd,  # +z-x sqrt of Y dagger
    'SqrtZ': gates_sq.S,    # +y+z sqrt of Z
    'SqrtZd': gates_sq.Sd,  # -y+z sqrt of Z dagger
}
