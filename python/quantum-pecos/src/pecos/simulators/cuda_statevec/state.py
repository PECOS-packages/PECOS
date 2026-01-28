# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""CUDA-accelerated state vector simulator using Rust cuQuantum bindings.

This module provides GPU-accelerated state vector simulation using NVIDIA cuQuantum
via the Rust pecos-cuquantum bindings (pecos-rslib-cuda).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pecos_rslib_cuda import CuStateVec as CuStateVecRs

from pecos.simulators.cuda_statevec import bindings
from pecos.simulators.sim_class_types import StateVector

if TYPE_CHECKING:
    import sys

    from pecos.typing import SimulatorGateParams

    # Handle Python 3.10 compatibility for Self type
    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing import TypeVar

        Self = TypeVar("Self", bound="CudaStateVec")


class CudaStateVec(StateVector):
    """GPU-accelerated state vector simulator using Rust cuQuantum bindings.

    This simulator uses NVIDIA cuQuantum SDK through Rust bindings for
    high-performance GPU-accelerated quantum simulation. It supports up to
    approximately 30 qubits (limited by GPU memory).

    Args:
        num_qubits: Number of qubits to simulate.
        seed: Optional random seed for reproducibility.

    Example:
        >>> sim = CudaStateVec(4)
        >>> sim.run_gate("H", [0])
        >>> sim.run_gate("CX", [(0, 1)])
        >>> result = sim.run_gate("Measure", [0])
    """

    def __init__(self, num_qubits: int, seed: int | None = None) -> None:
        """Initialize the CUDA state vector simulator.

        Args:
            num_qubits: Number of qubits to simulate.
            seed: Optional random seed for reproducibility.
        """
        if not isinstance(num_qubits, int):
            msg = "``num_qubits`` should be of type ``int``."
            raise TypeError(msg)

        super().__init__()

        self.bindings = bindings.gate_dict
        self.num_qubits = num_qubits

        # Create the Rust backend
        if seed is not None:
            self.backend = CuStateVecRs.with_seed(num_qubits, seed)
        else:
            self.backend = CuStateVecRs(num_qubits)

    def reset(self) -> Self:
        """Reset the quantum state to |0...0>."""
        self.backend.reset()
        return self

    def run_gate(
        self,
        symbol: str,
        locations: list | None = None,
        **params: SimulatorGateParams,
    ) -> dict:
        """Run a quantum gate operation.

        Args:
            symbol: The gate symbol (e.g., "H", "CX", "Measure").
            locations: List of qubit locations for the gate.
            **params: Additional parameters (e.g., angles for rotation gates).

        Returns:
            Dictionary of measurement results if the gate is a measurement,
            empty dictionary otherwise.
        """
        output = {}

        if params.get("simulate_gate", True) and locations:
            for location in locations:
                if symbol in self.bindings:
                    result = self.bindings[symbol](self, location, **params)
                    if result is not None:
                        output[location] = result

        return output

    def sample(self, num_samples: int) -> list[int]:
        """Sample measurement outcomes from the current state.

        This samples from the probability distribution of the current state
        without collapsing it.

        Args:
            num_samples: Number of samples to draw.

        Returns:
            List of bitstrings as integers. Each integer represents a measurement
            outcome where bit i corresponds to qubit i.
        """
        return self.backend.sample(num_samples)
