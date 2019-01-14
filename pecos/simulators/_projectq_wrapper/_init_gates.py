#   Copyright 2018 National Technology & Engineering Solutions of Sandia,
#   LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS,
#   the U.S. Government retains certain rights in this software.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from ._1q_gates import H2, H5, H6
from projectq.ops import H, X
from ._meas_gates import meas_z


def init_zero(state, qubit):
    """

    Args:
        state:
        qubit:

    Returns:

    """
    result = meas_z(state, qubit)

    if result:
        q = state.qids[qubit]
        X | q


def init_one(state, qubit):
    """
    Initialize qubit in state |1>.

    :param state:
    :param qubit:
    :return:
    """

    init_zero(state, qubit)
    q = state.qids[qubit]
    X | q


def init_plus(state, qubit):
    """
    Initialize qubit in state |+>.

    :param gens:
    :param qubit:
    :return:
    """

    init_zero(state, qubit)
    q = state.qids[qubit]
    H | q


def init_minus(state, qubit):
    """
    Initialize qubit in state |->

    :param gens:
    :param qubit:
    :return:
    """

    init_zero(state, qubit)
    q = state.qids[qubit]
    H2 | q


def init_plusi(state, qubit):
    """
    Initialize qubit in state |+i>

    :param gens:
    :param qubit:
    :return:
    """

    init_zero(state, qubit)
    q = state.qids[qubit]
    H5 | q


def init_minusi(state, qubit):
    """
    Initialize qubit in state |-i>

    Args:
        state:
        qubit:

    Returns:

    """

    init_zero(state, qubit)
    q = state.qids[qubit]
    H6 | q
