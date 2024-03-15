# Copyright 2022 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from typing import Tuple, Any
from numpy import pi
from ..cuconn import cq


def U1q(state, qubit: int, angles: Tuple[float, float], **params: Any) -> None:
    """
    U1q(theta, phi) = RZ(phi-pi/2)*RY(theta)*RZ(-phi+pi/2)

    Args:
        state:
        qubit:
        angles:
        **params:

    Returns:

    """

    theta, phi = angles
    g = cq.U1q(theta, phi)
    g.copy_to_device()
    g.apply(state.statevec, state.workspace, [], [qubit], False)
    g.free_on_device()


def RX(state, location, **params):
    angle = params["angle"]
    g = cq.Rx(angle)
    g.copy_to_device()
    g.apply(state.statevec, state.workspace, [], [location], False)
    g.free_on_device()


def RY(state, location, **params):
    angle = params["angle"]
    g = cq.Ry(angle)
    g.copy_to_device()
    g.apply(state.statevec, state.workspace, [], [location], False)
    g.free_on_device()


def RZ(state, location, **params):
    angle = params["angle"]
    g = cq.Rz(angle)
    g.copy_to_device()
    g.apply(state.statevec, state.workspace, [], [location], False)
    g.free_on_device()


def I(state, location, **params):
    pass


def X(state, location, **params):
    g = cq.PauliX()
    g.copy_to_device()
    g.apply(state.statevec, state.workspace, [], [location], False)
    g.free_on_device()


def Y(state, location, **params):
    g = cq.PauliY()
    g.copy_to_device()
    g.apply(state.statevec, state.workspace, [], [location], False)
    g.free_on_device()


def Z(state, location, **params):
    g = cq.PauliZ()
    g.copy_to_device()
    g.apply(state.statevec, state.workspace, [], [location], False)
    g.free_on_device()


def H(state, location, **params):
    g = cq.Hadamard()
    g.copy_to_device()
    g.apply(state.statevec, state.workspace, [], [location], False)
    g.free_on_device()


def Q(state, location, **params):
    g = cq.Rx(pi / 2)
    g.copy_to_device()
    g.apply(state.statevec, state.workspace, [], [location], False)
    g.free_on_device()


def Qd(state, location, **params):
    g = cq.Rx(-pi / 2)
    g.copy_to_device()
    g.apply(state.statevec, state.workspace, [], [location], False)
    g.free_on_device()


def R(state, location, **params):
    g = cq.Ry(pi / 2)
    g.copy_to_device()
    g.apply(state.statevec, state.workspace, [], [location], False)
    g.free_on_device()


def Rd(state, location, **params):
    g = cq.Ry(-pi / 2)
    g.copy_to_device()
    g.apply(state.statevec, state.workspace, [], [location], False)
    g.free_on_device()


def S(state, location, **params):
    g = cq.Rz(pi / 2)
    g.copy_to_device()
    g.apply(state.statevec, state.workspace, [], [location], False)
    g.free_on_device()


def Sd(state, location, **params):
    g = cq.Rz(-pi / 2)
    g.copy_to_device()
    g.apply(state.statevec, state.workspace, [], [location], False)
    g.free_on_device()
