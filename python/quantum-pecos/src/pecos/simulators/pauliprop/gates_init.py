# Copyright 2018 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Qubit initialization operations for Pauli fault propagation simulator.

This module provides quantum state initialization operations for the Pauli fault propagation simulator, including
functions to initialize qubits to computational basis states using efficient Pauli frame tracking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos.simulators.pauliprop.state import PauliProp
    from pecos.typing import SimulatorGateParams


def init(state: PauliProp, qubit: int, **_params: SimulatorGateParams) -> None:
    """Initialize qubit to zero state.

    Args:
        state: The PauliProp state instance.
        qubit (int): The qubit index to initialize.
        **_params: Unused additional parameters (kept for interface compatibility).
    """
    state.faults["X"].discard(qubit)
    state.faults["Y"].discard(qubit)
    state.faults["Z"].discard(qubit)
