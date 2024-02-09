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

from typing import Any

from pecos.simulators.sparsesim.cmd_meas import meas_z
from pecos.simulators.sparsesim.cmd_one_qubit import H2, H5, H6, H, X
from pecos.simulators.sparsesim.state import SparseSim


def init_zero(state: SparseSim, qubit: int, **params: Any) -> None:
    """

    Args:
        state: Instance representing the stabilizer state.
        qubit: Integer that indexes the qubit being acted on.
        **params:

    Returns: None

    """
    # Measure in the Z basis. (If random outcome, force a 0 outcome).
    # If outcome is 1 apply an X.
    if meas_z(state, qubit, forced_outcome=0):
        X(state, qubit)


def init_one(state: SparseSim, qubit: int, **params: Any) -> None:
    """Initialize qubit in state |1>.

    Args:
        state (SparseSim): Instance representing the stabilizer state.
        qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """
    if not meas_z(state, qubit, forced_outcome=1):
        X(state, qubit)


def init_plus(state: SparseSim, qubit: int, **params: Any) -> None:
    """Initialize qubit in state |+>.

    Args:
        state (SparseSim): Instance representing the stabilizer state.
        qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """
    init_zero(state, qubit)
    H(state, qubit)


def init_minus(state: SparseSim, qubit: int, **params: Any) -> None:
    """Initialize qubit in state |->.

    Args:
        state (SparseSim): Instance representing the stabilizer state.
        qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """
    init_zero(state, qubit)
    H2(state, qubit)


def init_plusi(state: SparseSim, qubit: int, **params: Any) -> None:
    """Initialize qubit in state |+i>.

    Args:
        state (SparseSim): Instance representing the stabilizer state.
        qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """
    init_zero(state, qubit)
    H5(state, qubit)


def init_minusi(state: SparseSim, qubit: int, **params: Any) -> None:
    """Initialize qubit in state |-i>.

    Args:
        state (SparseSim): Instance representing the stabilizer state.
        qubit (int): Integer that indexes the qubit being acted on.

    Returns: None

    """
    init_zero(state, qubit)
    H6(state, qubit)
