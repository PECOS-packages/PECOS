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
import numpy as np

def mat_paulix():
    return np.matrix([
        [0,1],
        [1,0]
        ], dtype=np.complex64)

def mat_pauliy():
    return np.matrix([
        [0,-1j],
        [1j,0]
        ], dtype=np.complex64)

def mat_pauliz():
    return np.matrix([
        [1,0],
        [0,-1]
        ], dtype=np.complex64)

def mat_hadamaard():
    return np.matrix([
        [np.sqrt(2),np.sqrt(2)],
        [np.sqrt(2),-np.sqrt(2)]
        ], dtype=np.complex64)

def mat_Rx(theta):
    return np.matrix([
        [np.cos(theta/2),        -1j*np.sin(theta/2)],
        [-1j*np.sin(theta/2), np.cos(theta/2)]
        ], dtype=np.complex64)

def mat_Ry(theta):
    return np.matrix([
        [np.cos(theta/2),        -np.sin(theta/2)],
        [np.sin(theta/2),     np.cos(theta/2)]
        ], dtype=np.complex64)

def mat_Rz(theta):
    return np.matrix([
        [np.cos(theta/2) - 1j*np.sin(theta/2),        0.0],
        [0.0,                                      np.cos(theta/2) + 1j*np.sin(theta/2)]
        ], dtype=np.complex64)

def U1q(state,
        location: int,
        angles: Tuple[float, float],
        **params: Any) -> None:
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

    # OLD CODE
    # g = cq.U1q(theta, phi)
    # g.copy_to_device()
    # g.apply(state.statevec, state.workspace, [], [qubit], False)
    # g.free_on_device()

    # CUQUANTUM STUFF
    U = mat_Rz(phi-pi/2) @ mat_Ry(theta) @ mat_Rz(-phi+pi/2)
    state.apply2x2matrix(location, U)

def RX(state, location, **params):

    angle = params['angle']

    # OLD CODE
    # g = cq.Rx(angle)
    # g.copy_to_device()
    # g.apply(state.statevec, state.workspace, [], [location], False)
    # g.free_on_device()

    # CUQUANTUM STUFF
    U = mat_Rx(angle)
    state.apply2x2matrix(location, U)


def RY(state, location, **params):

    angle = params['angle']

    # OLD CODE
    # g = cq.Ry(angle)
    # g.copy_to_device()
    # g.apply(state.statevec, state.workspace, [], [location], False)
    # g.free_on_device()

    # CUQUANTUM STUFF
    U = mat_Ry(angle)
    state.apply2x2matrix(location, U)

def RZ(state, location, **params):

    angle = params['angle']

    # OLD CODE
    # g = cq.Rz(angle)
    # g.copy_to_device()
    # g.apply(state.statevec, state.workspace, [], [location], False)
    # g.free_on_device()

    # CUQUANTUM STUFF
    U = mat_Rz(angle)
    state.apply2x2matrix(location, U)

def I(state, location, **params):
    pass


def X(state, location, **params):

    # OLD CODE
    # g = cq.PauliX()
    # g.copy_to_device()
    # g.apply(state.statevec, state.workspace, [], [location], False)
    # g.free_on_device()

    # CUQUANTUM STUFF
    U = mat_paulix()
    state.apply2x2matrix(location, U)

def Y(state, location, **params):

    # OLD CODE
    # g = cq.PauliY()
    # g.copy_to_device()
    # g.apply(state.statevec, state.workspace, [], [location], False)
    # g.free_on_device()

    # CUQUANTUM STUFF
    U = mat_pauliy()
    state.apply2x2matrix(location, U)


def Z(state, location, **params):

    # OLD CODE
    # g = cq.PauliZ()
    # g.copy_to_device()
    # g.apply(state.statevec, state.workspace, [], [location], False)
    # g.free_on_device()

    # CUQUANTUM STUFF
    U = mat_pauliz()
    state.apply2x2matrix(location, U)

def H(state, location, **params):

    # OLD CODE
    # g = cq.Hadamard()
    # g.copy_to_device()
    # g.apply(state.statevec, state.workspace, [], [location], False)
    # g.free_on_device()

    # CUQUANTUM STUFF
    U = mat_hadamaard()
    state.apply2x2matrix(location, U)

def Q(state, location, **params):

    # OLD CODE
    # g = cq.Rx(pi/2)
    # g.copy_to_device()
    # g.apply(state.statevec, state.workspace, [], [location], False)
    # g.free_on_device()

    RX(state, location, angle=pi/2)

def Qd(state, location, **params):

    # OLD CODE
    # g = cq.Rx(-pi/2)
    # g.copy_to_device()
    # g.apply(state.statevec, state.workspace, [], [location], False)
    # g.free_on_device()

    RX(state, location, angle=-pi/2)

def R(state, location, **params):

    # OLD CODE
    # g = cq.Ry(pi/2)
    # g.copy_to_device()
    # g.apply(state.statevec, state.workspace, [], [location], False)
    # g.free_on_device()

    RY(state, location, angle=pi/2)

def Rd(state, location, **params):

    # g = cq.Ry(-pi/2)
    # g.copy_to_device()
    # g.apply(state.statevec, state.workspace, [], [location], False)
    # g.free_on_device()

    RY(state, location, angle=-pi/2)

def S(state, location, **params):

    # g = cq.Rz(pi/2)
    # g.copy_to_device()
    # g.apply(state.statevec, state.workspace, [], [location], False)
    # g.free_on_device()

    RZ(state, location, angle=pi/2)

def Sd(state, location, **params):

    # g = cq.Rz(-pi/2)
    # g.copy_to_device()
    # g.apply(state.statevec, state.workspace, [], [location], False)
    # g.free_on_device()

    RZ(state, location, angle=-pi/2)