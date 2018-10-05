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
from ._1q_gates import Q, Qd, R, Rd, H2, H3, H4, H5, H6, F1, F1d, F2, F2d, F3, F3d, F4, F4d, I
from ._2q_gates import II, G2
from ._meas_gates import force_output, meas_z, meas_y, meas_x
from ._init_gates import init_zero, init_one, init_plus, init_minus, init_plusi, init_minusi

# Note: More ProjectQ gates can be added by updating the wrapper's `gate_dict` attribute.

# Note: Multiqubit gates are usually in all caps.

# TODO: Make a "MakeFunc" for measurements...


class MakeFunc:
    """
    Converts ProjectQ gate to a function.
    """
    def __init__(self, gate, angle=False):
        """

        Args:
            gate:
        """

        self.gate = gate
        self.angle = angle

    def func(self, state, qubits, **params):

        if isinstance(qubits, int):
            qs = state.qids[qubits]
        else:
            qs = []
            for q in qubits:
                qs.append(state.qids[q])

        if self.angle:
            self.gate(params['angle']) | qs
        else:
            self.gate | qs


gate_dict = {
    # ProjectQ specific:
    'T': MakeFunc(ops.T).func,  # fourth root of Z
    'Td': MakeFunc(ops.Tdag).func,  # fourth root of Z dagger
    'sqrtSWAP': MakeFunc(ops.SqrtSwap).func,
    'Entangle': MakeFunc(ops.Entangle).func,  # H on first qubit and CNOT to all others...
    'Rx': MakeFunc(ops.Rx, angle=True).func,  # Rotation about X (takes angle arg)
    'Ry': MakeFunc(ops.Ry, angle=True).func,  # Rotation about Y (takes angle arg)
    'Rz': MakeFunc(ops.Rz, angle=True).func,  # Rotation about Z (takes angle arg)
    'PhaseRot': MakeFunc(ops.R, angle=True).func,  # Phase-shift: Same as Rz but with a 1 in upper left of matrix.
    'TOFFOLI': MakeFunc(ops.Toffoli).func,
    'CRZ': MakeFunc(ops.CRz, angle=True).func,  # Controlled-Rz gate
    'CRX': MakeFunc(ops.C(ops.Rx, n=1), angle=True).func,  # Controlled-Rx
    'CRY': MakeFunc(ops.C(ops.Ry, n=1), angle=True).func,  # Controlled-Ry

    # Initialization
    # ==============
    'init |0>': init_zero,
    'init |1>': init_one,
    'init |+>': init_plus,
    'init |->': init_minus,
    'init |+i>': init_plusi,
    'init |-i>': init_minusi,

    # one-qubit operations
    # ====================

    # Paulis    # x->, z->
    'I': I,  # +x+z
    'X': MakeFunc(ops.X).func,  # +x-z
    'Y': MakeFunc(ops.Y).func,  # -x-z
    'Z': MakeFunc(ops.Z).func,  # -x+z

    # Square root of Paulis
    'Q': MakeFunc(Q).func,    # +x-y  sqrt of X
    'Qd': MakeFunc(Qd).func,  # +x+y sqrt of X dagger
    'R': MakeFunc(R).func,    # -z+x sqrt of Y
    'Rd': MakeFunc(Rd).func,  # +z-x sqrt of Y dagger
    'S': MakeFunc(ops.S).func,    # +y+z sqrt of Z
    'Sd': MakeFunc(ops.Sdag).func,  # -y+z sqrt of Z dagger

    # Hadamard-like
    'H': MakeFunc(ops.H).func,
    'H1': MakeFunc(ops.H).func,
    'H2': MakeFunc(H2).func,
    'H3': MakeFunc(H3).func,
    'H4': MakeFunc(H4).func,
    'H5': MakeFunc(H5).func,
    'H6': MakeFunc(H6).func,

    'H+z+x': MakeFunc(ops.H).func,
    'H-z-x': MakeFunc(H2).func,
    'H+y-z': MakeFunc(H3).func,
    'H-y-z': MakeFunc(H4).func,
    'H-x+y': MakeFunc(H5).func,
    'H-x-y': MakeFunc(H6).func,

    # Face rotations
    'F1': MakeFunc(F1).func,    # +y+x
    'F1d': MakeFunc(F1d).func,  # +z+y
    'F2': MakeFunc(F2).func,    # -z+y
    'F2d': MakeFunc(F2d).func,  # -y-x
    'F3': MakeFunc(F3).func,    # +y-x
    'F3d': MakeFunc(F3d).func,  # -z-y
    'F4': MakeFunc(F4).func,    # +z-y
    'F4d': MakeFunc(F4d).func,  # -y-z

    # two-qubit operations
    # ====================
    'CNOT': MakeFunc(ops.CNOT).func,
    'CZ': MakeFunc(ops.C(ops.Z)).func,
    'CY': MakeFunc(ops.C(ops.Y)).func,
    'SWAP': MakeFunc(ops.Swap).func,
    'G': G2,
    'II': II,

    # Measurements
    # ============
    'measure X': meas_x,
    'measure Y': meas_y,
    'measure Z': meas_z,
    'force output': force_output,
}
