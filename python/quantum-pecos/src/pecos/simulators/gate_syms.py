# Copyright 2023 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

alt_symbols = {
    # Initialization
    # ==============
    "Init": "Init +Z",
    "init |0>": "Init +Z",
    "init |1>": "Init -Z",
    "init |+>": "Init +X",
    "init |->": "Init -X",
    "init |+i>": "Init +Y",
    "init |-i>": "Init -Y",
    # one-qubit operations
    # ====================
    "RXY1Q": "R1XY",
    "U1q": "R1XY",
    # Square root of Paulis
    "Q": "SX",  # +x-y == R(X, pi/2)
    "Qd": "SXdg",  # +x+y == R(X, -pi/2)
    "R": "SY",  # -z+x == R(Y, pi/2)
    "Rd": "SYdg",  # +z-x == R(Y, -pi/2)
    "S": "SZ",  # +y+z == R(Z, pi/2)
    "Sd": "SZdg",  # -y+z == R(Z, -pi/2)
    "SqrtX": "SX",  # +x-y == R(X, pi/2)
    "SqrtXd": "SXdg",  # +x+y == R(X, -pi/2)
    "SqrtY": "SY",  # -z+x == R(Y, pi/2)
    "SqrtYd": "SYdg",  # +z-x == R(Y, -pi/2)
    "SqrtZ": "SZ",  # +y+z == R(Z, pi/2)
    "SqrtZd": "SZdg",  # -y+z == R(Z, -pi/2)
    # Hadamard-like
    "H1": "H",
    # 'H2': q1.H2,
    # 'H3': q1.H3,
    # 'H4': q1.H4,
    # 'H5': q1.H5,
    # 'H6': q1.H6,
    "H+z+x": "H",
    "H-z-x": "H2",
    "H+y-z": "H3",
    "H-y-z": "H4",
    "H-x+y": "H5",
    "H-x-y": "H6",
    # Face rotations
    "F1": "F",  # +y+x
    "F1d": "Fdg",  # +z+y
    "F2": "F2",  # -z+y
    "F2d": "F2dg",  # -y-x
    "F3": "F3",  # +y-x
    "F3d": "F3dg",  # -z-y
    "F4": "F4",  # +z-y
    "F4d": "F4dg",  # -y-z
    # two-qubit operations
    # ====================
    "CNOT": "CX",
    # 'G': q2.G2,
    # 'G2': q2.G2,
    # 'II': q2.II,
    "ZZPhase": "RZZ",
    "RXXYYZZ": "R2XXYYZZ",
    "ZZ": "SZZ",
    "ZZMax": "SZZ",
    # Mølmer-Sørensen gates
    "SqrtXX": "SXX",  # \equiv e^{+i (\pi /4)} * e^{-i (\pi /4) XX } == R(XX, pi/2)
    "SqrtYY": "SYY",
    "SqrtZZ": "SZZ",
    "SqrtXXd": "SXXdg",  # \equiv R(XX, -pi/2)
    "SqrtYYd": "SYYdg",
    "SqrtZZd": "SZZdg",
    "MS": "SXX",
    "MSXX": "SXX",
    "MSYY": "SYY",
    "MSZZ": "SZZ",
    # Measurements
    # ============
    "Measure": "Measure +Z",
    "measure X": "Measure +X",
    "measure Y": "Measure +Y",
    "measure Z": "Measure +Z",
    # 'force output': qmeas.force_output,
}
