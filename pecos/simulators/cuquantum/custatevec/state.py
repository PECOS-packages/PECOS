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
import cuquantum
from cuquantum import custatevec as cusv
from cuquantum import cudaDataType
try:
    from typing import Self
except:
    from typing_extensions import Self
from . import bindings


class CuStateVec:

    def __init__(self, num_qubits: int) -> None:

        if not isinstance(num_qubits, int):
            raise Exception('``num_qubits`` should be of type ``int.``')

        self.num_qubits = num_qubits

        # Set some things for cuquantum
        self._nIndexBits = num_qubits
        self._nSvSize = (1 << num_qubits)
        self._workspace = cusv.create()
        self._psi = None
        self._dtype = cudaDataType.CUDA_C_32F
        self._computetype = cuquantum.ComputeType.COMPUTE_32F
        
        # Initialize psi
        self.reset()
        
        self.workspace_gates = []
        self.bindings = bindings.gate_dict

    def reset(self) -> Self:
        """Reset the quantum state for another run without reinitializing."""

        del self._psi
        self._psi = cp.zeros(self._nSvSize, dtype=cp.complex64)
        self._psi[0] = 1

    def batch_measure(self, bit_ordering, randnum, collapse):
        bitOrdering  = np.asarray(bit_ordering, dtype=np.int32)
        bitStrings   = cp.empty(len(bit_ordering), dtype=cp.int32)

        if collapse:
            collapse_param = cusv.Collapse.NORMALIZE_AND_ZERO
        else:
            collapse_param = cusv.Collapse.NONE

        # batched measurement
        cusv.batch_measure(
            self._workspace, self._psi.data.ptr, self._dtype, self.num_qubits, 
            bitStrings.data.ptr, bitOrdering.ctypes.data, bitOrdering.size,
            randnum, collapse_param)

        return [int(i) for i in bitStrings]

    def get_probs(self) -> None:
        self.statevec.read_from_device()
        return self.statevec.get_probabilities(self.workspace)[0]

    def __del__(self):
        cusv.destroy(self._workspace)
        del self._workspace
        del self._psi

    def apply2x2matrix(self, q, U):
        # Check the inputs
        #  q: integer
        #  U: np.array([], dtype=np.complex64) -OR- cp.array([], dtype=np.complex64)

        if not isinstance(1, int):
            raise ValueError("q must be integer, found", q)

        if isinstance(U, np.ndarray):
            U_ptr = U.ctypes.data
        else:
            print(U)
            raise ValueError

        assert U.shape == (2,2), f"Invalid 2x2 unitary, shape={U.shape}"

        nTargets   = 1
        nControls  = 0
        adjoint    = 0

        targets    = np.asarray([q], dtype=np.int32)
        controls   = np.asarray([], dtype=np.int32)

        # cuStateVec handle initialization
        workspaceSize = cusv.apply_matrix_get_workspace_size(
            self._workspace, self._dtype, self._nIndexBits, U_ptr, self._dtype,
            cusv.MatrixLayout.ROW, adjoint, nTargets, nControls, self._computetype)

        # check the size of external workspace
        if workspaceSize > 0:
            workspace = cp.cuda.memory.alloc(workspaceSize)
            workspace_ptr = workspace.ptr
        else:
            workspace_ptr = 0

        # apply gate
        cusv.apply_matrix(
            self._workspace, self._psi.data.ptr, self._dtype, self._nIndexBits, U_ptr, self._dtype,
            cusv.MatrixLayout.ROW, adjoint, targets.ctypes.data, nTargets, controls.ctypes.data, 0, nControls,
            self._dtype, workspace_ptr, workspaceSize)

    def applyControlled2x2matrix(self, qc, qt, U):
        # Check the inputs
        #  q: integer
        #  U: np.array([], dtype=np.complex64) -OR- cp.array([], dtype=np.complex64)

        if not isinstance(1, int):
            raise ValueError("q must be integer, found", q)

        if isinstance(U, np.ndarray):
            U_ptr = U.ctypes.data
        else:
            print(U)
            raise ValueError

        assert U.shape == (2,2), f"Invalid 2x2 unitary, shape={U.shape}"

        nTargets   = 1
        nControls  = 1
        adjoint    = 0

        targets    = np.asarray([qt], dtype=np.int32)
        controls   = np.asarray([qc], dtype=np.int32)

        # cuStateVec handle initialization
        workspaceSize = cusv.apply_matrix_get_workspace_size(
            self._workspace, self._dtype, self._nIndexBits, U_ptr, self._dtype,
            cusv.MatrixLayout.ROW, adjoint, nTargets, nControls, self._computetype)

        # check the size of external workspace
        if workspaceSize > 0:
            workspace = cp.cuda.memory.alloc(workspaceSize)
            workspace_ptr = workspace.ptr
        else:
            workspace_ptr = 0

        # apply gate
        cusv.apply_matrix(
            self._workspace, self._psi.data.ptr, self._dtype, self._nIndexBits, U_ptr, self._dtype,
            cusv.MatrixLayout.ROW, adjoint, targets.ctypes.data, nTargets, controls.ctypes.data, 0, nControls,
            self._dtype, workspace_ptr, workspaceSize)

    def apply4x4matrix(self, q0, q1, U):
        # Check the inputs
        #  q: integer
        #  U: np.array([], dtype=np.complex64) -OR- cp.array([], dtype=np.complex64)

        if not isinstance(1, int):
            raise ValueError("q must be integer, found", q)

        if isinstance(U, np.ndarray):
            U_ptr = U.ctypes.data
        else:
            print(U)
            raise ValueError

        assert U.shape == (4,4), f"Invalid 4x4 unitary, shape={U.shape}"

        nTargets   = 2
        nControls  = 0
        adjoint    = 0

        targets    = np.asarray([q0, q1], dtype=np.int32)
        controls   = np.asarray([], dtype=np.int32)

        # cuStateVec handle initialization
        workspaceSize = cusv.apply_matrix_get_workspace_size(
            self._workspace, self._dtype, self._nIndexBits, U_ptr, self._dtype,
            cusv.MatrixLayout.ROW, adjoint, nTargets, nControls, self._computetype)

        # check the size of external workspace
        if workspaceSize > 0:
            workspace = cp.cuda.memory.alloc(workspaceSize)
            workspace_ptr = workspace.ptr
        else:
            workspace_ptr = 0

        # apply gate
        cusv.apply_matrix(
            self._workspace, self._psi.data.ptr, self._dtype, self._nIndexBits, U_ptr, self._dtype,
            cusv.MatrixLayout.ROW, adjoint, targets.ctypes.data, nTargets, controls.ctypes.data, 0, nControls,
            self._dtype, workspace_ptr, workspaceSize)