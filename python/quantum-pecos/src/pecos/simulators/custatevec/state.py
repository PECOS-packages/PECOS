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

import random

import cupy as cp
from cuquantum import ComputeType, cudaDataType
from cuquantum import custatevec as cusv
from numpy.typing import ArrayLike

from pecos.simulators.custatevec import bindings
from pecos.simulators.sim_class_types import StateVector


class CuStateVec(StateVector):
    """
    Simulation using cuQuantum's cuStateVec.
    """

    def __init__(self, num_qubits, seed=None) -> None:
        """
        Initializes the state vector.

        Args:
            num_qubits (int): Number of qubits being represented.
            seed (int): Seed for randomness.

        Returns:

        """

        if not isinstance(num_qubits, int):
            msg = "``num_qubits`` should be of type ``int``."
            raise TypeError(msg)
        random.seed(seed)

        super().__init__()

        self.bindings = bindings.gate_dict
        self.num_qubits = num_qubits

        # Set data type as double precision complex numbers
        self.cp_type = cp.complex128
        self.cuda_type = cudaDataType.CUDA_C_64F  # == cp.complex128
        self.compute_type = ComputeType.COMPUTE_64F

        # Allocate the statevector in GPU and initialize it to |0>
        self.reset()

        ####################################################
        # Set up cuStateVec library and GPU memory handles #
        ####################################################
        # All of this comes from:
        # https://github.com/NVIDIA/cuQuantum/blob/main/python/samples/custatevec/memory_handler.py

        # Check CUDA version and device config
        if cp.cuda.runtime.runtimeGetVersion() < 11020:
            msg = "CUDA 11.2+ is required."
            raise RuntimeError(
                msg,
            )
        dev = cp.cuda.Device()
        if not dev.attributes["MemoryPoolsSupported"]:
            msg = "Device does not support CUDA Memory pools."
            raise RuntimeError(
                msg,
            )

        # Avoid shrinking the pool
        mempool = cp.cuda.runtime.deviceGetDefaultMemPool(dev.id)
        cp.cuda.runtime.memPoolSetAttribute(
            mempool,
            cp.cuda.runtime.cudaMemPoolAttrReleaseThreshold,
            0xFFFFFFFFFFFFFFFF,  # = UINT64_MAX
        )

        # CuStateVec handle initialization
        self.libhandle = cusv.create()
        self.stream = cp.cuda.Stream()
        cusv.set_stream(self.libhandle, self.stream.ptr)

        # Device memory handler
        def malloc(size, stream):
            return cp.cuda.runtime.mallocAsync(size, stream)

        def free(ptr, size, stream):
            cp.cuda.runtime.freeAsync(ptr, stream)

        mem_handler = (malloc, free, "GPU memory handler")
        cusv.set_device_mem_handler(self.libhandle, mem_handler)

    def reset(self):
        """Reset the quantum state for another run without reinitializing."""
        # Initialize all qubits in the zero state
        self.cupy_vector = cp.zeros(shape=2**self.num_qubits, dtype=self.cp_type)
        self.cupy_vector[0] = 1
        return self

    def __del__(self) -> None:
        # CuPy will release GPU memory when the variable ``self.cupy_vector`` is no longer
        # reachable. However, we need to manually destroy the library handle.
        cusv.destroy(self.libhandle)

    @property
    def vector(self) -> ArrayLike:
        return self.cupy_vector.get()
