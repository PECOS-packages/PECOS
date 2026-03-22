"""Fake error model for testing and debugging.

This module provides a fake error model implementation that can be used
for testing quantum error correction systems without introducing actual
errors, useful for debugging and validation purposes.
"""

# Copyright 2020 The PECOS Developers
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

from pecos.noise.parent_class_error_gen import ParentErrorModel

if TYPE_CHECKING:
    from pecos.circuits.quantum_circuit import QuantumCircuit
    from pecos.noise.class_errors_circuit import ErrorCircuits
    from pecos.typing import ErrorParams, GateParams


class FakeErrorModel(ParentErrorModel):
    """Fake error model that returns pre-configured error circuits.

    This error model is designed for testing purposes and always returns
    the same pre-configured error circuits regardless of the input parameters.
    It provides a deterministic and controllable way to inject specific
    error patterns into quantum simulations.
    """

    def __init__(self, error_circuits: ErrorCircuits) -> None:
        """Initialize a FakeErrorModel with pre-configured error circuits.

        Args:
            error_circuits: Pre-configured ErrorCircuits instance to be returned
                for all error generation requests.
        """
        super().__init__()
        self.error_circuits = error_circuits
        self.leaked_qubits = set()

    def start(
        self,
        _circuit: QuantumCircuit,
        _error_params: ErrorParams,
    ) -> ErrorCircuits:
        """Start method that ignores parameters and returns pre-configured error circuits."""
        return self.error_circuits

    def generate_tick_errors(
        self,
        _tick_circuit: QuantumCircuit,
        _time: int | tuple[int, ...],
        _output: dict,
        **_params: GateParams,
    ) -> ErrorCircuits:
        """Generate tick errors that ignores parameters and returns pre-configured error circuits."""
        return self.error_circuits
