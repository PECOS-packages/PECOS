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

import numpy as np
import cupy as cp
from cuquantum import custatevec as cusv
from cuquantum import cudaDataType

from typing import Tuple, Any
from numpy import pi
import cuquantum

def mat_identity():
    return np.matrix([
        [1,0],
        [0,1]
        ], dtype=np.complex64)

def mat_paulix():
    return np.matrix([
        [0,1],
        [1,0]
        ], dtype=np.complex64)

def mat_sqrtzz():
    return np.matrix([
        [1, 0,  0,  0],
        [0, 1j, 0,  0],
        [0, 0,  1j, 0],
        [0, 0,  0,  1]
        ], dtype=np.complex64)

def mat_rzz(theta):
    a = np.exp(-1j*theta/2)
    b = np.exp(1j*theta/2)
    return np.matrix([
        [a, 0, 0, 0],
        [0, b, 0,  0],
        [0, 0, b, 0],
        [0, 0, 0, a]
        ], dtype=np.complex64)

def CX(state, location, **params):

    qc, qt = location
    # g = cq.PauliX()
    # g.copy_to_device()
    # g.apply(state.statevec, state.workspace, [qc], [qt], False)
    # g.free_on_device()

    # CUQUANTUM STUFF
    U = mat_paulix()
    state.applyControlled2x2matrix(qc, qt, U)


def SqrtZZ(state, location, **params):

    # g = cq.SqrtZZ()
    # g.copy_to_device()
    # g.apply(state.statevec, state.workspace, [], location, False)
    # g.free_on_device()

    # CUQUANTUM STUFF
    U = mat_sqrtzz()
    state.apply4x4matrix(location[0], location[1], U)


def RZZ(state, location, **params):

    angle = params['angle']
    # g = cq.RZZ(angle)
    # g.copy_to_device()
    # g.apply(state.statevec, state.workspace, [], location, False)
    # g.free_on_device()

    # CUQUANTUM STUFF
    U = mat_rzz(angle)
    state.apply4x4matrix(location[0], location[1], U)