# Copyright 2019 The PECOS Developers
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

from pecos.simulators.projectq.gates_meas import meas_z
from pecos.simulators.projectq.gates_one_qubit import H2, H5, H6, H, X


def init_zero(state, qubit, **params: Any):
    """Args:
    ----
        state:
        qubit:

    Returns:
    -------

    """
    result = meas_z(state, qubit)

    if result:
        X(state, qubit)


def init_one(state, qubit, **params: Any):
    """Initialize qubit in state |1>.

    :param state:
    :param qubit:
    :return:
    """
    init_zero(state, qubit)
    X(state, qubit)


def init_plus(state, qubit, **params: Any):
    """Initialize qubit in state |+>.

    :param gens:
    :param qubit:
    :return:
    """
    init_zero(state, qubit)
    H(state, qubit)


def init_minus(state, qubit, **params: Any):
    """Initialize qubit in state |->.

    :param gens:
    :param qubit:
    :return:
    """
    init_zero(state, qubit)
    H2(state, qubit)


def init_plusi(state, qubit, **params: Any):
    """Initialize qubit in state |+i>.

    :param gens:
    :param qubit:
    :return:
    """
    init_zero(state, qubit)
    H5(state, qubit)


def init_minusi(state, qubit, **params: Any):
    """Initialize qubit in state |-i>.

    Args:
    ----
        state:
        qubit:

    Returns:
    -------

    """
    init_zero(state, qubit)
    H6(state, qubit)
