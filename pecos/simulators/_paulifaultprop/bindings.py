# -*- coding: utf-8 -*-

#  =========================================================================  #
#   Copyright 2018 Ciarán Ryan-Anderson
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

"""
Specifies the symbol and function for each gate
"""
from . import gates_init, gates_one_qubit, gates_two_qubit, gates_meas

gate_dict = {
    # Initialization
    # ==============
    'init |0>': gates_init.init,
    'init |1>': gates_init.init,
    'init |+>': gates_init.init,
    'init |->': gates_init.init,
    'init |+i>': gates_init.init,
    'init |-i>': gates_init.init,

    # One-qubit Cliffords
    # ===================

    # Paulis
    'I': gates_one_qubit.I,
    'X': gates_one_qubit.X,
    'Y': gates_one_qubit.Y,
    'Z': gates_one_qubit.Z,

    # Square root of Paulis
    'Q': gates_one_qubit.Q,
    'Qd': gates_one_qubit.Qd,
    'R': gates_one_qubit.R,
    'Rd': gates_one_qubit.Rd,
    'S': gates_one_qubit.S,
    'Sd': gates_one_qubit.Sd,

    # Hadamard-like
    'H': gates_one_qubit.H,

    'H1': gates_one_qubit.H,
    'H2': gates_one_qubit.H2,
    'H3': gates_one_qubit.H3,
    'H4': gates_one_qubit.H4,
    'H5': gates_one_qubit.H5,
    'H6': gates_one_qubit.H6,

    # Face rotations
    'F1': gates_one_qubit.F1,    # +y+x
    'F1d': gates_one_qubit.F1d,  # +z+y
    'F2': gates_one_qubit.F2,    # -z+y
    'F2d': gates_one_qubit.F2d,  # -y-x
    'F3': gates_one_qubit.F3,    # +y-x
    'F3d': gates_one_qubit.F3d,  # -z-y
    'F4': gates_one_qubit.F4,    # +z-y
    'F4d': gates_one_qubit.F4d,  # -y-z

    # Two-qubit operations
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
    'measure X': gates_meas.meas_x,
    'measure Y': gates_meas.meas_y,
    'measure Z': gates_meas.meas_z,
    'measure Pauli': gates_meas.meas_pauli,
    'Check': gates_meas.meas_pauli,
    'force output': gates_meas.force_output,

}
