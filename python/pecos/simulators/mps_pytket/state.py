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

import random

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

    def __init__(self, num_qubits, seed=None) -> None:
        """
        Initializes the MPS.

        Args:
            num_qubits (int): Number of qubits being represented.
            seed (int): Seed for randomness.

        Returns:

        """

        if not isinstance(num_qubits, int):
            msg = "``num_qubits`` should be of type ``int``."
            raise TypeError(msg)
        random.seed(seed)  # TODO: This should instead be passed to Config

        super().__init__()

        self.bindings = bindings.gate_dict
        self.num_qubits = num_qubits

        # Configure the simulator
        # TODO: we might want to allow users to change some of these
        config = Config(
            chi=None,  # No truncation (exact simulation)
            truncation_fidelity=None,  # No truncation (exact simulation)
            float_precision=np.float64,  # np.float32 is also supported
            value_of_zero=1e-16,  # Values below this threshold are set to 0
            loglevel=30,  # Set to 10 for debug mode
        )
        self.dtype = config._complex_t  # noqa: SLF001

        # cuTensorNet handle initialization
        self.libhandle = CuTensorNetHandle()

        # Initialise the MPS on state |0>
        qubits = [Qubit(q) for q in range(num_qubits)]
        self.mps = MPSxGate(self.libhandle, qubits, config)

    def __del__(self) -> None:
        # CuPy will release GPU memory when the variable ``self.mps`` is no longer
        # reachable. However, we need to manually destroy the library handle.
        self.libhandle.destroy()
