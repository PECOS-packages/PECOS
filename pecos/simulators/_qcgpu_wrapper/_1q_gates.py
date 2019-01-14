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

import numpy as np
import qcgpu
from qcgpu.gate import x, y, z, sqrt_x, sqrt_y, s, h, t


class SingleQubit:

    def __init__(self, gate):

        if isinstance(gate, np.ndarray):
            self.gate = qcgpu.Gate(gate)
        else:
            self.gate = gate

    def func(self, state, qubit):
        state.apply_gate(self.gate, qubit)


T = SingleQubit(t()).func


def I(state, qubit):
    pass


X = SingleQubit(x()).func
Y = SingleQubit(y()).func
Z = SingleQubit(z()).func

Q = SingleQubit(sqrt_x()).func
Qd = SingleQubit(0.5*np.array([[1-1j, 1+1j], [1+1j, 1-1j]]))
R = SingleQubit(sqrt_y()).func
Rd = SingleQubit(0.5*np.array([[1-1j, 1-1j], [-1+1j, 1-1j]])).func
S = SingleQubit(s()).func
Sd = SingleQubit(0.5*np.array([[1-1j, 1-1j], [-1+1j, 1-1j]])).func

H = SingleQubit(h()).func
H2 = SingleQubit(0.5 * np.array([[1+1j, -1-1j], [-1-1j, -1-1j]])).func
H3 = SingleQubit(np.array([[0, 1], [1j, 0]])).func
H4 = SingleQubit(np.array([[0, 1j], [1, 0]])).func
H5 = SingleQubit(0.5 * np.array([[1+1j, 1-1j], [-1+1j, -1-1j]])).func
H6 = SingleQubit(0.5 * np.array([[-1-1j, 1-1j], [-1+1j, 1+1j]])).func

F1 = SingleQubit(0.5 * np.array([[1+1j, 1-1j], [1+1j, -1+1j]])).func
F1d = SingleQubit(0.5 * np.array([[1-1j, 1-1j], [1+1j, -1-1j]])).func
F2 = SingleQubit(0.5 * np.array([[1-1j, -1+1j], [1+1j, 1+1j]])).func
F2d = SingleQubit(0.5 * np.array([[1+1j, 1-1j], [-1-1j, 1-1j]])).func
F3 = SingleQubit(0.5 * np.array([[1-1j, 1+1j], [-1+1j, 1+1j]])).func
F3d = SingleQubit(0.5 * np.array([[1+1j, -1-1j], [1-1j, 1-1j]])).func
F4 = SingleQubit(0.5 * np.array([[1+1j, 1+1j], [1-1j, -1+1j]])).func
F4d = SingleQubit(0.5 * np.array([[1-1j, 1+1j], [1-1j, -1-1j]])).func
