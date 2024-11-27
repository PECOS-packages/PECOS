# Copyright 2024 The PECOS Developers
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
from pytket import Qubit
from pytket.extensions.cutensornet.structured_state import (
    Config,
    CuTensorNetHandle,
    MPSxGate,
)

from pecos.simulators.mps_pytket import bindings
from pecos.simulators.sim_class_types import StateTN


class MPS(StateTN):
    """
    Simulation using the gate-by-gate on demand MPS simulator from pytket-cutensornet.
    """

    def __init__(self, num_qubits, **mps_params) -> None:
        """
        Initializes the MPS.

        Args:
            num_qubits (int): Number of qubits being represented.
            mps_params: a collection of keyword arguments passed down to
                the ``Config`` object of the MPS. See the docs of pytket-cutensornet
                for a list of all available parameters.

        Returns:

        """

        if not isinstance(num_qubits, int):
            msg = "``num_qubits`` should be of type ``int``."
            raise TypeError(msg)

        super().__init__()

        self.bindings = bindings.gate_dict
        self.num_qubits = num_qubits

        # Configure the simulator
        self.config = Config(**mps_params)
        self.dtype = self.config._complex_t  # noqa: SLF001

        # cuTensorNet handle initialization
        self.libhandle = CuTensorNetHandle()

        # Initialise the MPS on state |0>
        self.reset()

    def reset(self) -> StateTN:
        """Reset the quantum state to all 0 for another run."""
        qubits = [Qubit(q) for q in range(self.num_qubits)]
        self.mps = MPSxGate(self.libhandle, qubits, self.config)
        self.mps._logger.info("Resetting MPS...")  # noqa: SLF001
        return self

    def __del__(self) -> None:
        # CuPy will release GPU memory when the variable ``self.mps`` is no longer
        # reachable. However, we need to manually destroy the library handle.
        self.libhandle.destroy()

    @property
    def vector(self) -> np.ndarray:
        """Obtain the statevector encoded in this MPS.

        Note:
            This is meant to be used for debugging only. Obtaining the statevector
            from the MPS on a large number of qubits completely defeats the purpose
            of tensor network methods.

        Returns:
            The statevector represented by the MPS as a numpy array.
        """
        return self.mps.get_statevector()
