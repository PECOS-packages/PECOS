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
import cuquantum
from cuquantum import custatevec as cusv
import cupy as cp


def apply2x2matrix(q, U):
    # Check the inputs
    #  q: integer
    #  U: np.array([], dtype=np.complex64) -OR- cp.array([], dtype=np.complex64)

    if not isinstance(1, int):
        raise ValueError("q must be integer, found", q)

    if isinstance(U, cp.ndarray):
        U_ptr = U.data.ptr
    elif isinstance(U, np.ndarray):
        U_ptr = U.ctypes.data
    else:
        raise ValueError

    assert U.shape == (2,2), "Invalid 2x2 unitary"

    nIndexBits = 3
    nSvSize    = (1 << nIndexBits)
    nTargets   = 1
    nControls  = 0
    adjoint    = 0

    targets    = np.asarray([q], dtype=np.int32)
    controls   = np.asarray([], dtype=np.int32)

    h_sv       = np.asarray([0.0+0.0j, 0.0+0.1j, 0.1+0.1j, 0.1+0.2j,
                            0.2+0.2j, 0.3+0.3j, 0.3+0.4j, 0.4+0.5j], dtype=np.complex64)

    d_sv = cp.asarray(h_sv)

    # cuStateVec handle initialization
    handle = cusv.create()
    workspaceSize = cusv.apply_matrix_get_workspace_size(
        handle, cuquantum.cudaDataType.CUDA_C_32F, nIndexBits, matrix_ptr, cuquantum.cudaDataType.CUDA_C_32F,
        cusv.MatrixLayout.ROW, adjoint, nTargets, nControls, cuquantum.ComputeType.COMPUTE_32F)

    # check the size of external workspace
    if workspaceSize > 0:
        workspace = cp.cuda.memory.alloc(workspaceSize)
        workspace_ptr = workspace.ptr
    else:
        workspace_ptr = 0

    # apply gate
    cusv.apply_matrix(
        handle, d_sv.data.ptr, cuquantum.cudaDataType.CUDA_C_32F, nIndexBits, matrix_ptr, cuquantum.cudaDataType.CUDA_C_32F,
        cusv.MatrixLayout.ROW, adjoint, targets.ctypes.data, nTargets, controls.ctypes.data, 0, nControls,
        cuquantum.ComputeType.COMPUTE_32F, workspace_ptr, workspaceSize)

    # destroy handle
    cusv.destroy(handle)

def U1q(state,
        qubit: int,
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
    cusv.apply_matrix(
        handle, d_sv.data.ptr, cuquantum.cudaDataType.CUDA_C_32F, nIndexBits, matrix_ptr, cuquantum.cudaDataType.CUDA_C_32F,
        cusv.MatrixLayout.ROW, adjoint, targets.ctypes.data, nTargets, controls.ctypes.data, 0, nControls,
        cuquantum.ComputeType.COMPUTE_32F, workspace_ptr, workspaceSize)

def RX(state, location, **params):

    # angle = params['angle']

    # OLD CODE
    # g = cq.Rx(angle)
    # g.copy_to_device()
    # g.apply(state.statevec, state.workspace, [], [location], False)
    # g.free_on_device()

    # CUQUANTUM STUFF
    cusv.apply_matrix(
        handle, d_sv.data.ptr, cuquantum.cudaDataType.CUDA_C_32F, nIndexBits, matrix_ptr, cuquantum.cudaDataType.CUDA_C_32F,
        cusv.MatrixLayout.ROW, adjoint, targets.ctypes.data, nTargets, controls.ctypes.data, 0, nControls,
        cuquantum.ComputeType.COMPUTE_32F, workspace_ptr, workspaceSize)


def RY(state, location, **params):

    angle = params['angle']

    # OLD CODE
    # g = cq.Ry(angle)
    # g.copy_to_device()
    # g.apply(state.statevec, state.workspace, [], [location], False)
    # g.free_on_device()

    # CUQUANTUM STUFF
    cusv.apply_matrix(
        handle, d_sv.data.ptr, cuquantum.cudaDataType.CUDA_C_32F, nIndexBits, matrix_ptr, cuquantum.cudaDataType.CUDA_C_32F,
        cusv.MatrixLayout.ROW, adjoint, targets.ctypes.data, nTargets, controls.ctypes.data, 0, nControls,
        cuquantum.ComputeType.COMPUTE_32F, workspace_ptr, workspaceSize)

def RZ(state, location, **params):

    angle = params['angle']

    # OLD CODE
    # g = cq.Rz(angle)
    # g.copy_to_device()
    # g.apply(state.statevec, state.workspace, [], [location], False)
    # g.free_on_device()

    # CUQUANTUM STUFF
    cusv.apply_matrix(
        handle, d_sv.data.ptr, cuquantum.cudaDataType.CUDA_C_32F, nIndexBits, matrix_ptr, cuquantum.cudaDataType.CUDA_C_32F,
        cusv.MatrixLayout.ROW, adjoint, targets.ctypes.data, nTargets, controls.ctypes.data, 0, nControls,
        cuquantum.ComputeType.COMPUTE_32F, workspace_ptr, workspaceSize)

def I(state, location, **params):
    pass


def X(state, location, **params):

    # OLD CODE
    # g = cq.PauliX()
    # g.copy_to_device()
    # g.apply(state.statevec, state.workspace, [], [location], False)
    # g.free_on_device()

    # CUQUANTUM STUFF
    cusv.apply_matrix(
        handle, d_sv.data.ptr, cuquantum.cudaDataType.CUDA_C_32F, nIndexBits, matrix_ptr, cuquantum.cudaDataType.CUDA_C_32F,
        cusv.MatrixLayout.ROW, adjoint, targets.ctypes.data, nTargets, controls.ctypes.data, 0, nControls,
        cuquantum.ComputeType.COMPUTE_32F, workspace_ptr, workspaceSize)

def Y(state, location, **params):

    # OLD CODE
    # g = cq.PauliY()
    # g.copy_to_device()
    # g.apply(state.statevec, state.workspace, [], [location], False)
    # g.free_on_device()


def Z(state, location, **params):

    # OLD CODE
    # g = cq.PauliZ()
    # g.copy_to_device()
    # g.apply(state.statevec, state.workspace, [], [location], False)
    # g.free_on_device()


def H(state, location, **params):

    # OLD CODE
    # g = cq.Hadamard()
    # g.copy_to_device()
    # g.apply(state.statevec, state.workspace, [], [location], False)
    # g.free_on_device()


def Q(state, location, **params):

    # OLD CODE
    # g = cq.Rx(pi/2)
    # g.copy_to_device()
    # g.apply(state.statevec, state.workspace, [], [location], False)
    # g.free_on_device()


def Qd(state, location, **params):

    # OLD CODE
    # g = cq.Rx(-pi/2)
    # g.copy_to_device()
    # g.apply(state.statevec, state.workspace, [], [location], False)
    # g.free_on_device()


def R(state, location, **params):

    # OLD CODE
    g = cq.Ry(pi/2)
    g.copy_to_device()
    g.apply(state.statevec, state.workspace, [], [location], False)
    g.free_on_device()


def Rd(state, location, **params):

    g = cq.Ry(-pi/2)
    g.copy_to_device()
    g.apply(state.statevec, state.workspace, [], [location], False)
    g.free_on_device()


def S(state, location, **params):

    g = cq.Rz(pi/2)
    g.copy_to_device()
    g.apply(state.statevec, state.workspace, [], [location], False)
    g.free_on_device()


def Sd(state, location, **params):

    g = cq.Rz(-pi/2)
    g.copy_to_device()
    g.apply(state.statevec, state.workspace, [], [location], False)
    g.free_on_device()