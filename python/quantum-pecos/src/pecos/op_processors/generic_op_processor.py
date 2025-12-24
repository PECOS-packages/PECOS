"""Generic operation processor for quantum error correction simulations.

This module provides a generic operation processor that handles the processing and transformation of quantum operations
within the PECOS quantum error correction framework, supporting various machine and error model protocols for flexible
quantum circuit execution and error injection.
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

from pecos.reps.pyphir import types as pt

if TYPE_CHECKING:
    from pecos.protocols import ErrorModelProtocol, MachineProtocol


class GenericOpProc:
    """Generic operation processor for quantum circuits.

    This class provides a generic framework for processing quantum operations
    with support for different machine architectures and error models.
    """

    def __init__(
        self,
        machine: MachineProtocol | None = None,
        error_model: ErrorModelProtocol | None = None,
    ) -> None:
        """Initialize the GenericOpProc.

        Args:
        ----
            machine: Optional machine protocol for processing machine operations.
            error_model: Optional error model protocol for applying noise.
        """
        self.machine = machine
        self.error_model = error_model

    def reset(self) -> None:
        """Reset state to initialization state."""

    def init(self) -> None:
        """Initialize the operation processor."""

    def attach_machine(self, machine: MachineProtocol) -> None:
        """Attach a machine to the operation processor.

        Args:
            machine: The machine protocol to attach.
        """
        self.machine = machine

    def attach_error_model(self, error_model: ErrorModelProtocol) -> None:
        """Attach an error model to the operation processor.

        Args:
            error_model: The error model protocol to attach.
        """
        self.error_model = error_model

    def shot_reinit(self) -> None:
        """Reinitialize for a new shot."""

    def process(self, buffered_ops: list) -> list:
        """Process buffered operations through machine and error model.

        Args:
            buffered_ops: List of operations to process.

        Returns:
            List of processed noisy quantum operations.
        """
        buffered_noisy_qops = []
        for op in buffered_ops:
            if isinstance(op, pt.opt.MOp):
                ops = self.machine.process([op])
                noisy_ops = self.error_model.process(ops)
            elif isinstance(op, pt.opt.QOp):
                noisy_ops = self.error_model.process([op])
            else:
                msg = f"This operation processor only handles MOps and QOps! Received type: {type(op)}"
                raise TypeError(msg)

            if noisy_ops:
                buffered_noisy_qops.extend(noisy_ops)

        return buffered_noisy_qops

    def process_meas(self, measurements: list[dict]) -> list[dict]:
        """Process measurement results.

        Args:
            measurements: List of measurement dictionaries.

        Returns:
            List of processed measurement dictionaries.
        """
        return measurements
