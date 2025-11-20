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

"""Quantum state representation for MPS PyTket simulator.

This module provides Matrix Product State quantum state representation and management for the PyTket-based simulator,
including tensor network storage and manipulation for low-entanglement quantum circuits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pytket import Qubit
from pytket.extensions.cutensornet.structured_state import (
    Config,
    CuTensorNetHandle,
    MPSxGate,
)

from pecos.simulators.mps_pytket import bindings
from pecos.simulators.sim_class_types import StateTN

if TYPE_CHECKING:
    from pecos import Array
    from pecos.typing import SimulatorInitParams


class MPS(StateTN):
    """Simulation using the gate-by-gate on demand MPS simulator from pytket-cutensornet."""

    def __init__(self, num_qubits: int, **mps_params: SimulatorInitParams) -> None:
        """Initializes the MPS.

        Args:
            num_qubits (int): Number of qubits being represented.
            mps_params: Configuration parameters passed to pytket-cutensornet's Config object.

        Keyword Args:
            chi (int | None): Maximum bond dimension. Lower values = faster but less accurate.
                Default: None (unlimited). For faster tests, try chi=16-64.
            truncation_fidelity (float | None): Target per-gate fidelity for SVD truncation.
                Lower values = faster but less accurate. Default: None (no truncation).
                For faster tests, try truncation_fidelity=0.999.
            kill_threshold (float): Threshold for discarding small singular values.
                Default: 0.0.
            seed (int | None): Random seed for sampling operations. Default: None.
            float_precision (type): Precision type (np.float32 or np.float64).
                Default: np.float64.
            value_of_zero (float): Value considered as zero. Default: 1e-16.
            leaf_size (int): Leaf size for internal algorithms. Default: 8.
            k (int): Parameter for internal algorithms. Default: 4.
            optim_delta (float): Optimization delta parameter. Default: 1e-5.
            loglevel (int): Logging level (10=DEBUG, 20=INFO, 30=WARNING).
                Default: 30 (WARNING).

        Note:
            For detailed documentation, see pytket-cutensornet Config class:
            https://docs.quantinuum.com/tket/extensions/pytket-cutensornet/
        """
        if not isinstance(num_qubits, int):
            msg = "``num_qubits`` should be of type ``int``."
            raise TypeError(msg)

        super().__init__()

        self.bindings = bindings.gate_dict
        self.num_qubits = num_qubits

        # Configure the simulator
        self.config = Config(**mps_params)
        self.dtype = self.config._complex_t

        # cuTensorNet handle initialization
        self.libhandle = CuTensorNetHandle()

        # Initialise the MPS on state |0>
        self.reset()

    def reset(self) -> StateTN:
        """Reset the quantum state to all 0 for another run."""
        qubits = [Qubit(q) for q in range(self.num_qubits)]
        self.mps = MPSxGate(self.libhandle, qubits, self.config)
        self.mps._logger.info("Resetting MPS...")
        return self

    def __del__(self) -> None:
        """Clean up tensor network library resources when the object is destroyed."""
        # CuPy will release GPU memory when the variable ``self.mps`` is no longer
        # reachable. However, we need to manually destroy the library handle.
        self.libhandle.destroy()

    @property
    def vector(self) -> Array:
        """Obtain the statevector encoded in this MPS.

        Note:
            This is meant to be used for debugging only. Obtaining the statevector
            from the MPS on a large number of qubits completely defeats the purpose
            of tensor network methods.

        Returns:
            The statevector represented by the MPS as a PECOS array.
        """
        return self.mps.get_statevector()
