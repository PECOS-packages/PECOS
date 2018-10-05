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

# import projectq.ops as ops
# from ._1q_gates import Q, Qd, R, Rd, H2, H3, H4, H5, H6, F1, F1d, F2, F2d, F3, F3d, F4, F4d, I
# from ._2q_gates import II, G2
# from ._meas_gates import force_output, meas_z, meas_y, meas_x
# from ._init_gates import init_zero, init_one, init_plus, init_minus, init_plusi, init_minusi
from cirq.ops import X, Y, Z

from cirq import ops
from ._1q_gates import I
from cirq.ops import CliffordGate

# Note: Multiqubit gates are usually in all caps.

# TODO: Make a "MakeFunc" for measurements...


class MakeFunc:
    """
    Converts cirq gate to a function.
    """
    def __init__(self, gate, angle=False):
        """

        Args:
            gate:
        """

        self.gate = gate
        self.angle = angle

    def func(self, state, qubits, **params):

        if self.angle:
            return self.gate(rads=params['angle'])(*qubits)
        else:
            return self.gate(*qubits)


gate_dict = {
    # ProjectQ specific:
    'T': MakeFunc(ops.T).func,  # fourth root of Z
    'Td': MakeFunc(ops.T.inverse()).func,  # fourth root of Z dagger
    'Rx': MakeFunc(ops.RotXGate, angle=True).func,  # Rotation about X (takes angle arg)
    'Ry': MakeFunc(ops.RotYGate, angle=True).func,  # Rotation about Y (takes angle arg)
    'Rz': MakeFunc(ops.RotZGate, angle=True).func,  # Rotation about Z (takes angle arg)
    'sqrtSWAP': MakeFunc(ops.SWAP ** 0.5).func,
    'CROT': MakeFunc(ops.Rot11Gate, angle=True).func,  # Puts a phase on |11> according to the angle.
    'TOFFOLI': MakeFunc(ops.TOFFOLI).func,
    'CCX': MakeFunc(ops.TOFFOLI).func,
    'CCZ': MakeFunc(ops.CCZ).func,
    'CSWAP': MakeFunc(ops.SWAP).func,
    'FREDKIN': MakeFunc(ops.FREDKIN).func,

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
    'Q': MakeFunc(ops.X**0.5).func,    # +x-y  sqrt of X
    'Qd': MakeFunc(ops.X**-0.5).func,  # +x+y sqrt of X dagger
    'R': MakeFunc(ops.Y**0.5).func,    # -z+x sqrt of Y
    'Rd': MakeFunc(ops.Y**-0.5).func,  # +z-x sqrt of Y dagger
    'S': MakeFunc(ops.S).func,    # +y+z sqrt of Z
    'Sd': MakeFunc(ops.S.inverse()).func,  # -y+z sqrt of Z dagger

    # Hadamard-like
    'H': MakeFunc(ops.H).func,
    'H1': MakeFunc(ops.H).func,
    'H2': MakeFunc(CliffordGate.from_xz_map((Z, True), (X, True))).func,
    'H3': MakeFunc(CliffordGate.from_xz_map((Y, False), (Z, True))).func,
    'H4': MakeFunc(CliffordGate.from_xz_map((Y, True), (Z, True))).func,
    'H5': MakeFunc(CliffordGate.from_xz_map((X, True), (Y, False))).func,
    'H6': MakeFunc(CliffordGate.from_xz_map((X, True), (Y, True))).func,

    'H+z+x': MakeFunc(ops.H).func,
    'H-z-x': MakeFunc(CliffordGate.from_xz_map((Z, True), (X, True))).func,
    'H+y-z': MakeFunc(CliffordGate.from_xz_map((Y, False), (Z, True))).func,
    'H-y-z': MakeFunc(CliffordGate.from_xz_map((Y, True), (Z, True))).func,
    'H-x+y': MakeFunc(CliffordGate.from_xz_map((X, True), (Y, False))).func,
    'H-x-y': MakeFunc(CliffordGate.from_xz_map((X, True), (Y, True))).func,

    # Face rotations
    'F1': MakeFunc(CliffordGate.from_xz_map((Y, False), (X, False))).func,    # +y+x
    'F1d': MakeFunc(CliffordGate.from_xz_map((Z, False), (Y, False))).func,  # +z+y
    'F2': MakeFunc(CliffordGate.from_xz_map((Z, True), (Y, False))).func,    # -z+y
    'F2d': MakeFunc(CliffordGate.from_xz_map((Y, True), (X, True))).func,  # -y-x
    'F3': MakeFunc(CliffordGate.from_xz_map((Y, False), (X, True))).func,    # +y-x
    'F3d': MakeFunc(CliffordGate.from_xz_map((Z, True), (Y, True))).func,  # -z-y
    'F4': MakeFunc(CliffordGate.from_xz_map((Z, False), (Y, True))).func,    # +z-y
    'F4d': MakeFunc(CliffordGate.from_xz_map((Y, True), (Z, True))).func,  # -y-z

    # two-qubit operations
    # ====================
    'CNOT': MakeFunc(ops.CNOT).func,
    'CZ': MakeFunc(ops.CZ).func,
    'CY': MakeFunc(ops.CY).func,
    'SWAP': MakeFunc(ops.SWAP).func,
    'G': G2,
    'II': II,

    # Measurements
    # ============
    'measure X': meas_x,
    'measure Y': meas_y,
    'measure Z': MakeFunc(ops.measure).func,
    'force output': force_output,
}
