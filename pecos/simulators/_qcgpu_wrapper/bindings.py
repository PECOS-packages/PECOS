#  =========================================================================  #
#   Copyright 2019 Ciarán Ryan-Anderson
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#  =========================================================================  #

from ._1q_gates import I, X, Y, Z, Q, Qd, R, Rd, S, Sd, T, H, H2, H3, H4, H5, H6, F1, F1d, F2, F2d,  F3, F3d, F4, F4d
from ._2q_gates import II, CNOT, CY, CZ, CS, CT, CH, SWAP, G2
from ._3q_gates import TOFFOLI, CCY, CCZ
from ._meas_gates import force_output, meas_x, meas_y, meas_z
from ._init_gates import init_zero, init_one, init_plus, init_minus, init_plusi, init_minusi

gate_dict = {
    'T': T,
    'CS': CS,
    'CT': CT,
    'CH': CH,
    'TOFFOLI': TOFFOLI,
    'CCX': TOFFOLI,
    'CCY': CCY,
    'CCZ': CCZ,

    # Initialization
    # ==============
    'init |0>': init_zero,     # Init by measuring (if entangle => random outcome)
    'init |1>': init_one,      # Init by measuring (if entangle => random outcome)
    'init |+>': init_plus,     # Init by measuring (if entangle => random outcome)
    'init |->': init_minus,    # Init by measuring (if entangle => random outcome)
    'init |+i>': init_plusi,   # Init by measuring (if entangle => random outcome)
    'init |-i>': init_minusi,  # Init by measuring (if entangle => random outcome)

    # one-qubit operations
    # ====================

    # Paulis    # x->, z->
    'I': I,  # +x+z
    'X': X,  # +x-z
    'Y': Y,  # -x-z
    'Z': Z,  # -x+z

    # Square root of Paulis
    'Q': Q,  # +x-y  sqrt of X
    'Qd': Qd,  # +x+y sqrt of X dagger
    'R': R,  # -z+x sqrt of Y
    'Rd': Rd,  # +z-x sqrt of Y dagger
    'S': S,  # +y+z sqrt of Z
    'Sd': Sd,  # -y+z sqrt of Z dagger

    # Hadamard-like
    'H': H,
    'H1': H,
    'H2': H2,
    'H3': H3,
    'H4': H4,
    'H5': H5,
    'H6': H6,

    'H+z+x': H,
    'H-z-x': H2,
    'H+y-z': H3,
    'H-y-z': H4,
    'H-x+y': H5,
    'H-x-y': H6,

    # Face rotations
    'F1': F1,  # +y+x
    'F1d': F1d,  # +z+y
    'F2': F2,  # -z+y
    'F2d': F2d,  # -y-x
    'F3': F3,  # +y-x
    'F3d': F3d,  # -z-y
    'F4': F4,  # +z-y
    'F4d': F4d,  # -y-z

    # two-qubit operations
    # ====================
    'CNOT': CNOT,
    'CX': CNOT,
    'CY': CY,
    'CZ': CZ,
    'SWAP': SWAP,
    'G': G2,
    'G2': G2,
    'II': II,
    # 'Sqrt XX': None,
    # 'MS': None,
    # 'MS XX': None,

    # Measurements
    # ============
    'measure X': meas_x,  # no random_output (force outcome) !
    'measure Y': meas_y,  # no random_output (force outcome) !
    'measure Z': meas_z,  # no random_output (force outcome) !
    'force output': force_output,
}
