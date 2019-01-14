#  =========================================================================  #
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
#  =========================================================================  #

from .cmd_meas import meas_z
from .cmd_one_qubit import H, H2, H5, H6, X


def init_zero(state, location):
    """

    Args:
        state:
        location:

    Returns:

    """
    result = meas_z(state, location, random_outcome=0)

    if result:
        X(state, location)


def init_one(gens, qubit):
    """
    Initialize qubit in state |1>.

    :param gens:
    :param qubit:
    :return:
    """

    init_zero(gens, qubit)
    X(gens, qubit)


def init_plus(gens, qubit):
    """
    Initialize qubit in state |+>.

    :param gens:
    :param qubit:
    :return:
    """

    init_zero(gens, qubit)
    H(gens, qubit)


def init_minus(gens, qubit):
    """
    Initialize qubit in state |->

    :param gens:
    :param qubit:
    :return:
    """

    init_zero(gens, qubit)
    H2(gens, qubit)


def init_plusi(gens, qubit):
    """
    Initialize qubit in state |+i>

    :param gens:
    :param qubit:
    :return:
    """

    init_zero(gens, qubit)
    H5(gens, qubit)


def init_minusi(gens, qubit):
    """
    Initialize qubit in state |-i>

    :param gens:
    :param qubit:
    :return:
    """

    init_zero(gens, qubit)
    H6(gens, qubit)
