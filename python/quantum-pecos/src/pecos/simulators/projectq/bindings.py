# Copyright 2018 The PECOS Developers
# Copyright 2018 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract
# DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from projectq import ops

from pecos.simulators.projectq import (
    gates_init,
    gates_meas,
    gates_one_qubit,
    gates_two_qubit,
)
from pecos.simulators.projectq.helper import MakeFunc

# Note: More ProjectQ gates can be added by updating the wrapper's `gate_dict` attribute.

# Note: Multiqubit gates are usually in all caps.


gate_dict = {
    # ProjectQ specific:
    "T": MakeFunc(ops.T).func,  # fourth root of Z
    "Tdg": MakeFunc(ops.Tdag).func,  # fourth root of Z dagger
    "SSWAP": MakeFunc(ops.SqrtSwap).func,
    "Entangle": MakeFunc(
        ops.Entangle,
    ).func,  # H on first qubit and CNOT to all others...
    "RX": gates_one_qubit.RX,  # Rotation about X (takes angle arg)
    "RY": gates_one_qubit.RY,  # Rotation about Y (takes angle arg)
    "RZ": gates_one_qubit.RZ,  # Rotation about Z (takes angle arg)
    "R1XY": gates_one_qubit.R1XY,
    "PhaseRot": MakeFunc(
        ops.R,
        angle=True,
    ).func,  # Phase-shift: Same as Rz but with a 1 in upper left of matrix.
    "TOFFOLI": MakeFunc(ops.Toffoli).func,
    "CRZ": MakeFunc(ops.CRz, angle=True).func,  # Controlled-Rz gate
    "CRX": MakeFunc(ops.C(ops.Rx, 1), angle=True).func,  # Controlled-Rx
    "CRY": MakeFunc(ops.C(ops.Ry, 1), angle=True).func,  # Controlled-Ry
    "RXX": gates_two_qubit.RXX,
    "RYY": gates_two_qubit.RYY,
    "RZZ": gates_two_qubit.RZZ,
    "R2XXYYZZ": gates_two_qubit.R2XXYYZZ,
    # Initialization
    # ==============
    "Init +Z": gates_init.init_zero,  # Init by measuring (if entangle => random outcome
    "Init -Z": gates_init.init_one,  # Init by measuring (if entangle => random outcome
    "Init +X": gates_init.init_plus,  # Init by measuring (if entangle => random outcome
    "Init -X": gates_init.init_minus,  # Init by measuring (if entangle => random outcome
    "Init +Y": gates_init.init_plusi,  # Init by measuring (if entangle => random outcome
    "Init -Y": gates_init.init_minusi,  # Init by measuring (if entangle => random outcome
    "leak": gates_init.init_zero,
    "leak |0>": gates_init.init_zero,
    "leak |1>": gates_init.init_one,
    "unleak |0>": gates_init.init_zero,
    "unleak |1>": gates_init.init_one,
    # one-qubit operations
    # ====================
    # Paulis    # x->, z->
    "I": gates_one_qubit.Identity,  # +x+z
    "X": gates_one_qubit.X,  # +x-z
    "Y": gates_one_qubit.Y,  # -x-z
    "Z": gates_one_qubit.Z,  # -x+z
    # Square root of Paulis
    "SX": gates_one_qubit.SX,  # +x-y  sqrt of X
    "SXdg": gates_one_qubit.SXdg,  # +x+y sqrt of X dagger
    "SY": gates_one_qubit.SY,  # -z+x sqrt of Y
    "SYdg": gates_one_qubit.SYdg,  # +z-x sqrt of Y dagger
    "SZ": gates_one_qubit.SZ,  # +y+z sqrt of Z
    "SZdg": gates_one_qubit.SZdg,  # -y+z sqrt of Z dagger
    # Hadamard-like
    "H": gates_one_qubit.H,
    "H2": gates_one_qubit.H2,
    "H3": gates_one_qubit.H3,
    "H4": gates_one_qubit.H4,
    "H5": gates_one_qubit.H5,
    "H6": gates_one_qubit.H6,
    # Face rotations
    "F": gates_one_qubit.F,  # +y+x
    "Fdg": gates_one_qubit.Fdg,  # +z+y
    "F2": gates_one_qubit.F2,  # -z+y
    "F2dg": gates_one_qubit.F2dg,  # -y-x
    "F3": gates_one_qubit.F3,  # +y-x
    "F3dg": gates_one_qubit.F3dg,  # -z-y
    "F4": gates_one_qubit.F4,  # +z-y
    "F4dg": gates_one_qubit.F4dg,  # -y-z
    # two-qubit operations
    # ====================
    "CX": gates_two_qubit.CNOT,
    "CY": gates_two_qubit.CY,
    "CZ": gates_two_qubit.CZ,
    "SWAP": gates_two_qubit.SWAP,
    "G": gates_two_qubit.G2,
    "G2": gates_two_qubit.G2,
    "II": gates_two_qubit.II,
    # Mølmer-Sørensen gates
    "SXX": gates_two_qubit.SXX,  # \equiv e^{+i (\pi /4)} * e^{-i (\pi /4) XX } == R(XX, pi/2)
    "SYY": gates_two_qubit.SYY,
    "SZZ": gates_two_qubit.SZZ,
    # Measurements
    # ============
    "Measure +X": gates_meas.meas_x,  # no random_output (force outcome) !
    "Measure +Y": gates_meas.meas_y,  # no random_output (force outcome) !
    "Measure +Z": gates_meas.meas_z,  # no random_output (force outcome) !
    "force output": gates_meas.force_output,
}
