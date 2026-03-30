# Copyright 2018 The PECOS Developers
# Copyright 2018 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract
# DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Qubit initialization commands for sparse stabilizer simulator.

This module provides quantum state initialization operations for the sparse stabilizer simulator, including
commands to initialize qubits to computational basis states using efficient stabilizer tableau representation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos.simulators.sparsesim.state import SparseStabPy
    from pecos.typing import SimulatorGateParams

from pecos.simulators.sparsesim.cmd_meas import meas_z
from pecos.simulators.sparsesim.cmd_one_qubit import H2, H5, H6, H, X


def init_zero(
    state: SparseStabPy,
    qubit: int,
    forced_outcome: int = -1,
    **_params: SimulatorGateParams,
) -> None:
    """Initialize qubit to zero state.

    Args:
        state (SparseStabPy): Instance representing the stabilizer state.
        qubit (int): Integer that indexes the qubit being acted on.
        forced_outcome (int): Value for a "random" outcome. Default is -1, which means 0 and 1 are equally probable.
        **_params: Unused additional parameters (kept for interface compatibility).
    """
    # Measure in the Z basis. (If random outcome, force a 0 outcome).
    # If outcome is 1 apply an X.
    if meas_z(state, qubit, forced_outcome=forced_outcome):
        X(state, qubit)


def init_one(
    state: SparseStabPy,
    qubit: int,
    forced_outcome: int = -1,
    **_params: SimulatorGateParams,
) -> None:
    """Initialize qubit in state |1>.

    Args:
        state (SparseStabPy): Instance representing the stabilizer state.
        qubit (int): Integer that indexes the qubit being acted on.
        forced_outcome: Value for a "random" outcome. Default is -1, which means 0 and 1 are equally probable.
    """
    if not meas_z(state, qubit, forced_outcome=forced_outcome):
        X(state, qubit)


def init_plus(
    state: SparseStabPy,
    qubit: int,
    forced_outcome: int = -1,
    **_params: SimulatorGateParams,
) -> None:
    """Initialize qubit in state |+>.

    Args:
        state (SparseStabPy): Instance representing the stabilizer state.
        qubit (int): Integer that indexes the qubit being acted on.
        forced_outcome: Value for a "random" outcome. Default is -1, which means 0 and 1 are equally probable.
    """
    init_zero(state, qubit, forced_outcome=forced_outcome)
    H(state, qubit)


def init_minus(
    state: SparseStabPy,
    qubit: int,
    forced_outcome: int = -1,
    **_params: SimulatorGateParams,
) -> None:
    """Initialize qubit in state |->.

    Args:
        state (SparseStabPy): Instance representing the stabilizer state.
        qubit (int): Integer that indexes the qubit being acted on.
        forced_outcome: Value for a "random" outcome. Default is -1, which means 0 and 1 are equally probable.
    """
    init_zero(state, qubit, forced_outcome=forced_outcome)
    H2(state, qubit)


def init_plusi(
    state: SparseStabPy,
    qubit: int,
    forced_outcome: int = -1,
    **_params: SimulatorGateParams,
) -> None:
    """Initialize qubit in state |+i>.

    Args:
        state (SparseStabPy): Instance representing the stabilizer state.
        qubit (int): Integer that indexes the qubit being acted on.
        forced_outcome: Value for a "random" outcome. Default is -1, which means 0 and 1 are equally probable.
    """
    init_zero(state, qubit, forced_outcome=forced_outcome)
    H5(state, qubit)


def init_minusi(
    state: SparseStabPy,
    qubit: int,
    forced_outcome: int = -1,
    **_params: SimulatorGateParams,
) -> None:
    """Initialize qubit in state |-i>.

    Args:
        state (SparseStabPy): Instance representing the stabilizer state.
        qubit (int): Integer that indexes the qubit being acted on.
        forced_outcome: Value for a "random" outcome. Default is -1, which means 0 and 1 are equally probable.
    """
    init_zero(state, qubit, forced_outcome=forced_outcome)
    H6(state, qubit)
