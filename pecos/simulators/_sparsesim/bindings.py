#  =========================================================================  #
#   Copyright 2018 National Technology & Engineering Solutions of Sandia,
#   LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS,
#   the U.S. Government retains certain rights in this software.
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

from . import cmd_one_qubit as q1
from . import cmd_two_qubit as q2
from . import cmd_init as qinit
from . import cmd_meas as qmeas

gate_dict = {
    # Initialization
    # ==============
    'init |0>': qinit.init_zero,
    'init |1>': qinit.init_one,
    'init |+>': qinit.init_plus,
    'init |->': qinit.init_minus,
    'init |+i>': qinit.init_plusi,
    'init |-i>': qinit.init_minusi,

    # circuit element symbol to function

    # one-qubit operations
    # ====================

    # Paulis    # x->, z->
    'I': q1.I,  # +x+z
    'X': q1.X,  # +x-z
    'Y': q1.Y,  # -x-z
    'Z': q1.Z,  # -x+z

    # Square root of Paulis
    'Q': q1.Q,    # +x-y
    'Qd': q1.Qd,  # +x+y
    'R': q1.R,    # -z+x
    'Rd': q1.Rd,  # +z-x
    'S': q1.S,    # +y+z
    'Sd': q1.Sd,  # -y+z

    # Hadamard-like
    'H': q1.H,

    'H1': q1.H,
    'H2': q1.H2,
    'H3': q1.H3,
    'H4': q1.H4,
    'H5': q1.H5,
    'H6': q1.H6,

    'H+z+x': q1.H,
    'H-z-x': q1.H2,
    'H+y-z': q1.H3,
    'H-y-z': q1.H4,
    'H-x+y': q1.H5,
    'H-x-y': q1.H6,

    # Face rotations
    'F1': q1.F1,    # +y+x
    'F1d': q1.F1d,  # +z+y
    'F2': q1.F2,    # -z+y
    'F2d': q1.F2d,  # -y-x
    'F3': q1.F3,    # +y-x
    'F3d': q1.F3d,  # -z-y
    'F4': q1.F4,    # +z-y
    'F4d': q1.F4d,  # -y-z

    # two-qubit operations
    # ====================
    'CNOT': q2.cnot,
    'CZ': q2.CZ,
    'CY': q2.CY,
    'SWAP': q2.SWAP,
    'G': q2.G2,
    'II': q2.II,

    # Measurements
    # ============
    'measure X': qmeas.meas_x,
    'measure Y': qmeas.meas_y,
    'measure Z': qmeas.meas_z,
    'force output': qmeas.force_output,
}

'''
sym2func = {
    # circuit element symbol to function

    # one-qubit operations
    # ====================

    # Paulis
    # ......
    'idle': op_1.I,
    'noop': op_1.I,
    'I': op_1.I,
    'X': op_1.X,
    'Y': op_1.Y,
    'Z': op_1.Z,

    # Square root of Paulis
    # .....................
    'Q': op_1.Q,
    'R': op_1.R,
    'S': op_1.S,

    # Complex transpose of the square root of Paulis
    # ..............................................
    'Qd': op_1.Qd,
    'Rd': op_1.Rd,
    'Sd': op_1.Sd,

    # Hadamard-like operations
    # ........................
    'H': op_1.H,
    'H1': op_1.H,
    'H2': op_1.H2,
    'H3': op_1.H3,
    'H4': op_1.H4,
    'H5': op_1.H5,
    'H6': op_1.H6,

    # Hadamard-like operations
    # ........................
    'Hx+z': op_1.H,
    'Hz+y': op_1.H2,
    'Hy+x': op_1.H3,
    'Hx-z': op_1.H4,
    'Hz-y': op_1.H5,
    'Hy-x': op_1.H6,


    # Rotations around the faces of the stabilizer octahedron
    # .......................................................
    'F1': op_1.F1,
    'F2': op_1.F2,
    'F3': op_1.F3,
    'F4': op_1.F4,

    'F1d': op_1.F1d,
    'F2d': op_1.F2d,
    'F3d': op_1.F3d,
    'F4d': op_1.F4d,

    # Rotations around the faces of the stabilizer octahedron
    # .......................................................
    'Fxyz': op_1.F1,
    'F-xzy': op_1.F2,
    'Fxy-z': op_1.F3,
    'Fxz-y': op_1.F4,

    'Fxzy': op_1.F1d,
    'F-xyz': op_1.F2d,
    'Fx-zy': op_1.F3d,
    'Fx-yz': op_1.F4d,

    # initialization
    # ..............
    'init |0>': op_i.init_zero,
    'init |1>': op_i.init_one,
    'init |+>': op_i.init_plus,
    'init |->': op_i.init_minus,
    'init |+i>': op_i.init_plusi,
    'init |-i>': op_i.init_minusi,

    # two-qubit operations
    # ====================
    'CNOT': op_2.CNOT,
    'SWAP': op_2.SWAP,
    'CZ': op_2.CZ,
    'G2': op_2.G2,

    # Measurements
    # ============
    'measure X': op_m.meas_x,
    'measure Y': op_m.meas_y,
    'measure Z': op_m.meas_z,

    # 'ideal measure X': op_i.init_plus,
    # 'ideal measure Y': op_i.init_plusi,
    'ideal measure Z': op_m.meas_z_ideal,
}
'''