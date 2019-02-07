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

import numpy as np
from projectq.ops import BasicGate, get_inverse


def I(state, qubit, **kwargs):
    """
    Identity, which does nothing.

    :param gens:
    :param qubit:
    :return:
    """
    pass


class QGate(BasicGate):
    """Square-root of X gate class"""
    @property
    def matrix(self):
        return 0.5 * np.array([[1+1j, 1-1j], [1-1j, 1+1j]])

    def text_str(self):
        return r'$\sqrt{X}'

    def __str__(self):
        return "Q"


Q = QGate()
Qd = get_inverse(Q)


class RGate(BasicGate):
    """Square-root of Y gate class"""
    @property
    def matrix(self):
        return 0.5 * np.array([[1+1j, -1-1j], [1+1j, 1+1j]])

    def text_str(self):
        return r'$\sqrt{Y}'

    def __str__(self):
        return "R"


R = RGate()
Rd = get_inverse(R)


class H2Gate(BasicGate):
    """Hadmard-like gate #2"""
    @property
    def matrix(self):
        return 0.5 * np.array([[1+1j, -1-1j], [-1-1j, -1-1j]])

    def __str__(self):
        return "H2"


H2 = H2Gate()


class H3Gate(BasicGate):
    """Hadmard-like gate #3"""
    @property
    def matrix(self):
        return np.array([[0, 1], [1j, 0]])

    def __str__(self):
        return "H3"


H3 = H3Gate()


class H4Gate(BasicGate):
    """Hadmard-like gate #4"""
    @property
    def matrix(self):
        return np.array([[0, 1j], [1, 0]])

    def __str__(self):
        return "H4"


H4 = H4Gate()


class H5Gate(BasicGate):
    """Hadmard-like gate #5"""
    @property
    def matrix(self):
        return 0.5 * np.array([[1+1j, 1-1j], [-1+1j, -1-1j]])

    def __str__(self):
        return "H5"


H5 = H5Gate()


class H6Gate(BasicGate):
    """Hadmard-like gate #6"""
    @property
    def matrix(self):
        return 0.5 * np.array([[-1-1j, 1-1j], [-1+1j, 1+1j]])

    def __str__(self):
        return "H6"


H6 = H6Gate()


class F1Gate(BasicGate):
    """Face rotations of an octahedron #1"""
    @property
    def matrix(self):
        return 0.5 * np.array([[1+1j, 1-1j], [1+1j, -1+1j]])

    def __str__(self):
        return "F1"


F1 = F1Gate()
F1d = get_inverse(F1)


class F2Gate(BasicGate):
    """Face rotations of an octahedron #2"""
    @property
    def matrix(self):
        return 0.5 * np.array([[1-1j, -1+1j], [1+1j, 1+1j]])

    def __str__(self):
        return "F2"


F2 = F2Gate()
F2d = get_inverse(F2)


class F3Gate(BasicGate):
    """Face rotations of an octahedron #3"""
    @property
    def matrix(self):
        return 0.5 * np.array([[1-1j, 1+1j], [-1+1j, 1+1j]])

    def __str__(self):
        return "F3"


F3 = F3Gate()
F3d = get_inverse(F3)


class F4Gate(BasicGate):
    """Face rotations of an octahedron #3"""
    @property
    def matrix(self):
        return 0.5 * np.array([[1+1j, 1+1j], [1-1j, -1+1j]])

    def __str__(self):
        return "F4"


F4 = F3Gate()
F4d = get_inverse(F4)
