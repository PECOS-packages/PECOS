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

"""Quantum state representation for cuStateVec simulator.

This module provides GPU-accelerated quantum state representation and management for the NVIDIA cuStateVec simulator,
including CUDA-based state vector storage and manipulation for high-performance quantum simulation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
from cuquantum import ComputeType, cudaDataType
from cuquantum.bindings import custatevec as cusv

from pecos.simulators.custatevec import bindings
from pecos.simulators.sim_class_types import StateVector

if TYPE_CHECKING:
    import sys

    from pecos import Array

    # Handle Python 3.10 compatibility for Self type
    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing import TypeVar

        Self = TypeVar("Self", bound="CuStateVec")


class CuStateVec(StateVector):
    """Simulation using cuQuantum's cuStateVec."""

    def __init__(self, num_qubits: int, _seed: int | None = None) -> None:
        """Initializes the state vector.

        Args:
            num_qubits (int): Number of qubits being represented.
            _seed (int): Seed for randomness (kept for API compatibility, not used in GPU-based simulator).
        """
        self.libhandle = None
        if not isinstance(num_qubits, int):
            msg = "``num_qubits`` should be of type ``int``."
            raise TypeError(msg)

        super().__init__()

        self.bindings = bindings.gate_dict
        self.num_qubits = num_qubits

        # Set data type as double precision complex numbers
        self.cp_type = cp.complex128
        self.cuda_type = cudaDataType.CUDA_C_64F  # == cp.complex128
        self.compute_type = ComputeType.COMPUTE_64F

        # Allocate the statevector in GPU and initialize it to |0>
        self.cupy_vector = None
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
        def malloc(
            size: int,
            stream: object,
        ) -> int:  # stream: CUDA stream object (opaque type)
            return cp.cuda.runtime.mallocAsync(size, stream)

        def free(
            ptr: int,
            _size: int,
            stream: object,
        ) -> None:  # stream: CUDA stream object (opaque type)
            cp.cuda.runtime.freeAsync(ptr, stream)

        mem_handler = (malloc, free, "GPU memory handler")
        cusv.set_device_mem_handler(self.libhandle, mem_handler)

    def reset(self) -> Self:
        """Reset the quantum state for another run without reinitializing."""
        # Initialize all qubits in the zero state
        if self.cupy_vector is not None:
            self.cupy_vector[:] = 0
            self.cupy_vector[0] = 1
        else:
            self.cupy_vector = cp.zeros(shape=2**self.num_qubits, dtype=self.cp_type)
            self.cupy_vector[0] = 1
        return self

    def __del__(self) -> None:
        """Clean up GPU resources when the object is destroyed."""
        # CuPy will release GPU memory when the variable ``self.cupy_vector`` is no longer
        # reachable. However, we need to manually destroy the library handle.
        if self.libhandle:
            cusv.destroy(self.libhandle)

    @property
    def vector(self) -> Array:
        """Get the quantum state vector from GPU memory.

        Returns:
            The state vector transferred from GPU to CPU memory.
        """
        return self.cupy_vector.get()
