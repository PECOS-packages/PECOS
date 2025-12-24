"""Base error model implementation framework.

This module provides the foundation for error model implementations in PECOS,
including base classes and interfaces for creating custom error models
for quantum error correction simulations.
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

from pecos.reps.pyphir.op_types import EMOp, MOp, QOp

if TYPE_CHECKING:
    from collections.abc import Callable

    from pecos.protocols import MachineProtocol


class NoErrorModel:
    """Represents having no error model."""

    def __init__(self) -> None:
        """Initialize a NoErrorModel instance.

        Creates an empty error model that applies no noise to quantum operations.
        """
        self.error_params = {}
        self.machine = None
        self.num_qubits = None

    def reset(self) -> None:
        """Reset state to initialization state."""

    def init(self, num_qubits: int, machine: MachineProtocol | None = None) -> None:
        """Initialize the error model.

        Args:
            num_qubits: Number of qubits in the system.
            machine: Optional machine protocol for hardware-specific behavior.

        Raises:
            Exception: If error parameters are provided for a no-error model.
        """
        self.machine = machine
        self.num_qubits = num_qubits
        if self.error_params:
            msg = "No error model is being utilized but error parameters are being provided!"
            raise Exception(msg)

    def shot_reinit(self) -> None:
        """Reinitialize for each shot (no-op for base error model)."""

    def process(
        self,
        ops: list,
        _call_back: Callable | None = None,
    ) -> list | None:
        """Process operations without applying any errors.

        Args:
            ops: List of operations to process.
            call_back: Optional callback function (unused).

        Returns:
            List of processed operations without errors.
        """
        noisy_ops = []
        for op in ops:
            if isinstance(op, QOp):
                noisy_ops.append(op)
            elif isinstance(op, MOp | EMOp):
                pass
            else:
                msg = f"Operation type '{type(op)}' is not supported!"
                raise NotImplementedError(msg)
        return noisy_ops
