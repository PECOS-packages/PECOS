# Copyright 2018 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Specifies the symbol and function for each gate."""

from pecos.simulators.paulifaultprop import (
    gates_init,
    gates_meas,
    gates_one_qubit,
    gates_two_qubit,
)

gate_dict = {
    # Initialization
    # ==============
    # just removes errors:
    "init |0>": gates_init.init,
    "init |1>": gates_init.init,
    "init |+>": gates_init.init,
    "init |->": gates_init.init,
    "init |+i>": gates_init.init,
    "init |-i>": gates_init.init,
    # One-qubit Cliffords
    # ===================
    # Paulis
    "I": gates_one_qubit.Identity,
    "X": gates_one_qubit.X,
    "Y": gates_one_qubit.Y,
    "Z": gates_one_qubit.Z,
    # Square root of Paulis
    "SX": gates_one_qubit.SX,
    "SXdg": gates_one_qubit.SXdg,
    "SY": gates_one_qubit.SY,
    "SYdg": gates_one_qubit.SYdg,
    "SZ": gates_one_qubit.SZ,
    "SZdg": gates_one_qubit.SZdg,
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
    # Two-qubit operations
    # ====================
    "CX": gates_two_qubit.CX,
    "CZ": gates_two_qubit.CZ,
    "CY": gates_two_qubit.CY,
    "SWAP": gates_two_qubit.SWAP,
    "G": gates_two_qubit.G2,
    "G2": gates_two_qubit.G2,
    "II": gates_two_qubit.II,
    # Mølmer-Sørensen gates
    "SqrtXX": gates_two_qubit.SXX,  # \equiv e^{+i (\pi /4)} * e^{-i (\pi /4) XX } == R(XX, pi/2)
    "SqrtZZ": gates_two_qubit.SZZ,
    "MS": gates_two_qubit.SXX,
    "MSXX": gates_two_qubit.SXX,
    # Measurements
    # ============
    "measure X": gates_meas.meas_x,
    "measure Y": gates_meas.meas_y,
    "measure Z": gates_meas.meas_z,
    # Measure general operators (here... just some Pauli)
    "measure": gates_meas.meas_pauli,
    "check": gates_meas.meas_pauli,
    # TODO: all simulators should have this... and should measure as general as possible
    "force output": gates_meas.force_output,
}
