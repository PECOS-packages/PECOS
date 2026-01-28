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

"""CUDA-accelerated stabilizer simulator using Rust cuQuantum bindings.

This module provides GPU-accelerated stabilizer simulation using NVIDIA cuQuantum
via the Rust pecos-cuquantum bindings (pecos-rslib-cuda).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pecos_rslib_cuda import CuStabilizer as CuStabilizerRs

from pecos.simulators.cuda_stabilizer import bindings
from pecos.simulators.sim_class_types import Stabilizer

if TYPE_CHECKING:
    import sys

    from pecos.typing import SimulatorGateParams

    # Handle Python 3.10 compatibility for Self type
    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing import TypeVar

        Self = TypeVar("Self", bound="CudaStabilizer")


class CudaStabilizer(Stabilizer):
    """GPU-accelerated stabilizer simulator using Rust cuQuantum bindings.

    This simulator uses NVIDIA cuQuantum SDK through Rust bindings for
    high-performance GPU-accelerated stabilizer simulation. It can handle
    thousands of qubits efficiently but only supports Clifford gates.

    Note:
        This simulator only supports Clifford gates. Non-Clifford gates
        (T gates, arbitrary rotations) are not available. For universal
        quantum simulation, use CudaStateVec instead.

    Args:
        num_qubits: Number of qubits to simulate.
        seed: Optional random seed for reproducibility.

    Example:
        >>> sim = CudaStabilizer(100)
        >>> sim.run_gate("H", [0])
        >>> sim.run_gate("CX", [(0, 1)])
        >>> result = sim.run_gate("Measure", [0])
    """

    def __init__(self, num_qubits: int, seed: int | None = None) -> None:
        """Initialize the CUDA stabilizer simulator.

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
            self.backend = CuStabilizerRs.with_seed(num_qubits, seed)
        else:
            self.backend = CuStabilizerRs(num_qubits)

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
            **params: Additional parameters.

        Returns:
            Dictionary of measurement results if the gate is a measurement,
            empty dictionary otherwise.

        Raises:
            ValueError: If a non-Clifford gate is requested.
        """
        output = {}

        if params.get("simulate_gate", True) and locations:
            for location in locations:
                if symbol in self.bindings:
                    result = self.bindings[symbol](self, location, **params)
                    if result is not None:
                        output[location] = result
                elif symbol in (
                    "T",
                    "Tdg",
                    "RX",
                    "RY",
                    "RZ",
                    "R1XY",
                    "RXX",
                    "RYY",
                    "RZZ",
                ):
                    msg = (
                        f"Gate '{symbol}' is not a Clifford gate and is not supported by "
                        "CudaStabilizer. Use CudaStateVec for non-Clifford gates."
                    )
                    raise ValueError(msg)

        return output
