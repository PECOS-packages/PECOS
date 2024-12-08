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

from pecos.simulators.sparsesim import cmd_init as qinit
from pecos.simulators.sparsesim import cmd_meas as qmeas
from pecos.simulators.sparsesim import cmd_one_qubit as q1
from pecos.simulators.sparsesim import cmd_two_qubit as q2

gate_dict = {
    # Initialization
    # ==============
    "Init +Z": qinit.init_zero,
    "Init -Z": qinit.init_one,
    "Init +X": qinit.init_plus,
    "Init -X": qinit.init_minus,
    "Init +Y": qinit.init_plusi,
    "Init -Y": qinit.init_minusi,
    "leak": qinit.init_one,
    "leak |0>": qinit.init_zero,
    "leak |1>": qinit.init_one,
    "unleak |0>": qinit.init_zero,
    "unleak |1>": qinit.init_one,
    # circuit element symbol to function
    # one-qubit operations
    # ====================
    # Paulis    # x->, z->
    "I": q1.Identity,  # +x+z == R(U, 0)
    "X": q1.X,  # +x-z == R(X, pi)
    "Y": q1.Y,  # -x-z == R(Y, pi)
    "Z": q1.Z,  # -x+z == R(Z, pi)
    # Square root of Paulis
    "SX": q1.SX,  # +x-y == R(X, pi/2)
    "SXdg": q1.SXdg,  # +x+y == R(X, -pi/2)
    "SY": q1.SY,  # -z+x == R(Y, pi/2)
    "SYdg": q1.SYdg,  # +z-x == R(Y, -pi/2)
    "SZ": q1.SZ,  # +y+z == R(Z, pi/2)
    "SZdg": q1.SZdg,  # -y+z == R(Z, -pi/2)
    # Hadamard-like
    "H": q1.H,
    "H2": q1.H2,
    "H3": q1.H3,
    "H4": q1.H4,
    "H5": q1.H5,
    "H6": q1.H6,
    # Face rotations
    "F": q1.F,  # +y+x
    "Fdg": q1.Fdg,  # +z+y
    "F2": q1.F2,  # -z+y
    "F2dg": q1.F2dg,  # -y-x
    "F3": q1.F3,  # +y-x
    "F3dg": q1.F3dg,  # -z-y
    "F4": q1.F4,  # +z-y
    "F4dg": q1.F4dg,  # -y-z
    # two-qubit operations
    # ====================
    "CX": q2.CX,
    "CY": q2.CY,
    "CZ": q2.CZ,
    "SWAP": q2.SWAP,
    "G": q2.G2,
    "G2": q2.G2,
    "II": q2.II,
    # Mølmer-Sørensen gates
    "SXX": q2.SXX,  # \equiv e^{+i (\pi /4)} * e^{-i (\pi /4) XX } == R(XX, pi/2)
    "SYY": q2.SYY,
    "SZZ": q2.SZZ,
    "SXXdg": q2.SXXdg,
    "SYYdg": q2.SYYdg,
    "SZZdg": q2.SZZdg,
    # Measurements
    # ============
    "Measure +X": qmeas.meas_x,
    "Measure +Y": qmeas.meas_y,
    "Measure +Z": qmeas.meas_z,
    "force output": qmeas.force_output,
}
