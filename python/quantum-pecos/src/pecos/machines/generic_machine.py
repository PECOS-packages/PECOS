"""Generic machine implementation for quantum error correction simulations.

This module provides a generic machine implementation that serves as a foundation for quantum hardware and simulator
abstractions within the PECOS framework, enabling consistent interfaces for quantum operation execution across
different physical and virtual quantum computing platforms.
"""

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

from __future__ import annotations

from typing import TYPE_CHECKING

from pecos.reps.pyphir.op_types import QOp

if TYPE_CHECKING:
    from pecos.reps.pyphir.op_types import MOp


class GenericMachine:
    """Represents generic, abstract machine."""

    def __init__(
        self,
        machine_params: dict | None = None,
        num_qubits: int | None = None,
        metadata: dict | None = None,
        pos: dict | None = None,
    ) -> None:
        """Initialize a GenericMachine.

        Args:
        ----
            machine_params: Optional dictionary of machine-specific parameters.
            num_qubits: Number of qubits in the machine.
            metadata: Optional metadata dictionary.
            pos: Optional position information for qubits.
        """
        self.machine_params = (
            dict(machine_params) if machine_params is not None else None
        )
        self.num_qubits = num_qubits
        self.metadata = metadata
        self.pos = pos
        self.qubit_set = set(range(num_qubits)) if num_qubits is not None else set()
        self.leaked_qubits = set()
        self.lost_qubits = set()

    def reset(self) -> None:
        """Reset state to initialization state."""
        self.leaked_qubits.clear()
        self.lost_qubits.clear()

    def init(self, num_qubits: int | None = None) -> None:
        """Initialize the machine with the specified number of qubits.

        Args:
            num_qubits: Number of qubits to initialize.
        """
        self.num_qubits = num_qubits
        self.qubit_set = set(range(num_qubits))

    def shot_reinit(self) -> None:
        """Reinitialize for a new shot by resetting state."""
        self.reset()

    def process(self, op_buffer: list[QOp | MOp]) -> list:
        """Process a buffer of quantum and machine operations.

        Args:
            op_buffer: List of quantum or machine operations to process.

        Returns:
            List of processed operations.
        """
        for op in op_buffer:
            if "mop" in op.name:
                print("MOP >", op)

        return op_buffer

    def leak(self, qubits: set[int]) -> list[QOp]:
        """Starts tracking qubits as leaked qubits and calls the quantum simulation appropriately to trigger leakage."""
        self.leaked_qubits |= qubits
        return [QOp(name="Init", args=list(qubits))]

    def _unleak(self, qubits: set[int]) -> list[QOp]:
        """Untrack qubits as leaked qubits and calls the quantum simulation appropriately to trigger leakage."""
        self.leaked_qubits -= qubits
        return []

    def unleak(self, qubits: set[int]) -> list[QOp]:
        """Untrack qubits as leaked qubits and calls the quantum simulation appropriately to trigger leakage."""
        self._unleak(qubits)
        return []

    def unleak_to_zero(self, qubits: set[int]) -> list[QOp]:
        """Unleak qubits and initialize them to the |0⟩ state.

        Args:
            qubits: Set of qubit indices to unleak and initialize.

        Returns:
            List of quantum operations to initialize qubits to |0⟩.
        """
        self._unleak(qubits)
        return [
            QOp(name="Init +Z", args=list(qubits)),
        ]

    def unleak_to_one(self, qubits: set[int]) -> list[QOp]:
        """Unleak qubits and initialize them to the |1⟩ state.

        Args:
            qubits: Set of qubit indices to unleak and initialize.

        Returns:
            List of quantum operations to initialize qubits to |1⟩.
        """
        self._unleak(qubits)
        return [
            QOp(name="Init -Z", args=list(qubits)),
        ]

    def meas_leaked(self, qubits: set[int]) -> list[QOp]:
        """Measure leaked qubits and remove them from leaked tracking.

        Args:
            qubits: Set of qubit indices to measure.

        Returns:
            List of measurement operations for the leaked qubits.
        """
        self.leaked_qubits -= qubits
        return [
            QOp(name="Init -Z", args=list(qubits)),
        ]
