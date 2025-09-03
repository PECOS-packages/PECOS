# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Measurement operations for Qulacs simulator.

This module provides quantum measurement operations for the Qulacs simulator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos.simulators.qulacs import Qulacs
    from pecos.typing import SimulatorGateParams


def meas_z(state: Qulacs, qubit: int, **_params: SimulatorGateParams) -> int:
    """Measure qubit in Z basis.

    Args:
        state: An instance of Qulacs
        qubit: The index of the qubit to measure

    Returns:
        The measurement outcome (0 or 1)
    """
    result = state.qulacs_state.run_1q_gate("MZ", qubit)
    return int(result) if result is not None else 0
