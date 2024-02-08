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

import projectq.ops as ops
from .helper import MakeFunc
from . import gates_init, gates_one_qubit, gates_two_qubit, gates_meas

# Note: More ProjectQ gates can be added by updating the wrapper's `gate_dict` attribute.

# Note: Multiqubit gates are usually in all caps.


gate_dict = {
    # ProjectQ specific:
    'T': MakeFunc(ops.T).func,  # fourth root of Z
    'Td': MakeFunc(ops.Tdag).func,  # fourth root of Z dagger
    'sqrtSWAP': MakeFunc(ops.SqrtSwap).func,
    'Entangle': MakeFunc(ops.Entangle).func,  # H on first qubit and CNOT to all others...
    'RX': MakeFunc(ops.Rx, angle=True).func,  # Rotation about X (takes angle arg)
    'RY': MakeFunc(ops.Ry, angle=True).func,  # Rotation about Y (takes angle arg)
    'RZ': MakeFunc(ops.Rz, angle=True).func,  # Rotation about Z (takes angle arg)
    'PhaseRot': MakeFunc(ops.R, angle=True).func,  # Phase-shift: Same as Rz but with a 1 in upper left of matrix.
    'TOFFOLI': MakeFunc(ops.Toffoli).func,
    'CRZ': MakeFunc(ops.CRz, angle=True).func,  # Controlled-Rz gate
    'CRX': MakeFunc(ops.C(ops.Rx, 1), angle=True).func,  # Controlled-Rx
    'CRY': MakeFunc(ops.C(ops.Ry, 1), angle=True).func,  # Controlled-Ry

    'RXX': gates_two_qubit.RXX,
    'RYY': gates_two_qubit.RYY,
    'RZZ': gates_two_qubit.RZZ,

    # Initialization
    # ==============
    'init |0>': gates_init.init_zero,  # Init by measuring (if entangle => random outcome
    'init |1>': gates_init.init_one,  # Init by measuring (if entangle => random outcome
    'init |+>': gates_init.init_plus,  # Init by measuring (if entangle => random outcome
    'init |->': gates_init.init_minus,  # Init by measuring (if entangle => random outcome
    'init |+i>': gates_init.init_plusi,  # Init by measuring (if entangle => random outcome
    'init |-i>': gates_init.init_minusi,  # Init by measuring (if entangle => random outcome

    # one-qubit operations
    # ====================

    # Paulis    # x->, z->
    'I': gates_one_qubit.I,  # +x+z
    'X': gates_one_qubit.X,  # +x-z
    'Y': gates_one_qubit.Y,  # -x-z
    'Z': gates_one_qubit.Z,  # -x+z

    # Square root of Paulis
    'Q': gates_one_qubit.Q,    # +x-y  sqrt of X
    'Qd': gates_one_qubit.Qd,  # +x+y sqrt of X dagger
    'R': gates_one_qubit.R,    # -z+x sqrt of Y
    'Rd': gates_one_qubit.Rd,  # +z-x sqrt of Y dagger
    'S': gates_one_qubit.S,    # +y+z sqrt of Z
    'Sd': gates_one_qubit.Sd,  # -y+z sqrt of Z dagger

    # Hadamard-like
    'H': gates_one_qubit.H,
    'H1': gates_one_qubit.H,
    'H2': gates_one_qubit.H2,
    'H3': gates_one_qubit.H3,
    'H4': gates_one_qubit.H4,
    'H5': gates_one_qubit.H5,
    'H6': gates_one_qubit.H6,

    'H+z+x': gates_one_qubit.H,
    'H-z-x': gates_one_qubit.H2,
    'H+y-z': gates_one_qubit.H3,
    'H-y-z': gates_one_qubit.H4,
    'H-x+y': gates_one_qubit.H5,
    'H-x-y': gates_one_qubit.H6,

    # Face rotations
    'F1': gates_one_qubit.F1,    # +y+x
    'F1d': gates_one_qubit.F1d,  # +z+y
    'F2': gates_one_qubit.F2,    # -z+y
    'F2d': gates_one_qubit.F2d,  # -y-x
    'F3': gates_one_qubit.F3,    # +y-x
    'F3d': gates_one_qubit.F3d,  # -z-y
    'F4': gates_one_qubit.F4,    # +z-y
    'F4d': gates_one_qubit.F4d,  # -y-z

    # two-qubit operations
    # ====================
    'CNOT': gates_two_qubit.CNOT,
    'CZ': gates_two_qubit.CZ,
    'CY': gates_two_qubit.CY,
    'SWAP': gates_two_qubit.SWAP,
    'G': gates_two_qubit.G2,
    'G2': gates_two_qubit.G2,
    'II': gates_two_qubit.II,

    # Mølmer–Sørensen gates
    'SqrtXX': gates_two_qubit.SqrtXX,  # \equiv e^{+i (\pi /4)} * e^{-i (\pi /4) XX } == R(XX, pi/2)
    'MS': gates_two_qubit.SqrtXX,
    'MSXX': gates_two_qubit.SqrtXX,

    # Measurements
    # ============
    'measure X': gates_meas.meas_x,  # no random_output (force outcome) !
    'measure Y': gates_meas.meas_y,  # no random_output (force outcome) !
    'measure Z': gates_meas.meas_z,  # no random_output (force outcome) !
    'force output': gates_meas.force_output,
}
