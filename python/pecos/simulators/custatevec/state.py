# TODO: Include license information?

import random

import cupy as cp
from cuquantum import ComputeType, cudaDataType
from cuquantum import custatevec as cusv

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
        # cusv.logger_set_level(5)  # Remove! here for debugging

        self.bindings = bindings.gate_dict
        self.num_qubits = num_qubits

        # Set data type as double precision complex numbers
        self.cp_type = cp.complex128
        self.cuda_type = cudaDataType.CUDA_C_64F  # == cp.complex128
        self.compute_type = ComputeType.COMPUTE_64F

        # Allocate the statevector in GPU and initialize it to |0>
        self.vector = cp.zeros(shape=2**num_qubits, dtype=self.cp_type)
        self.vector[0] = 1

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

    def __del__(self) -> None:
        # CuPy will release GPU memory when the variable ``self.vector`` is no longer
        # reachable. However, we need to manually destroy the library handle.
        cusv.destroy(self.libhandle)
