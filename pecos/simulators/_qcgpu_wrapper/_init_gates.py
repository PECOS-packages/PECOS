#  =========================================================================  #
#   Copyright 2019 Ciarán Ryan-Anderson
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

from ._1q_gates import H5, H6


def init_zero(state, qubit):
    """
    Initialize qubit in state |0>

    Args:
        state:
        qubit:

    Returns:

    """

    result = state.measure_collapse(qubit)

    if result == "1":
        state.x(qubit)


def init_one(state, qubit):
    """
    Initialize qubit in state |1>

    Args:
        state:
        qubit:

    Returns:

    """

    result = state.measure_collapse(qubit)

    if result == "0":
        state.x(qubit)


def init_plus(state, qubit):
    """
    Initialize qubit in state |+>

    Args:
        state:
        qubit:

    Returns:

    """

    result = state.measure_collapse(qubit)

    if result == "1":
        state.x(qubit)

    state.h(qubit)


def init_minus(state, qubit):
    """
    Initialize qubit in state |->

    Args:
        state:
        qubit:

    Returns:

    """

    result = state.measure_collapse(qubit)

    if result == "0":
        state.x(qubit)

    state.h(qubit)


def init_plusi(state, qubit):
    """
    Initialize qubit in state |+i>

    Args:
        state:
        qubit:

    Returns:

    """

    result = state.measure_collapse(qubit)

    if result == "1":
        state.x(qubit)

    H5(state, qubit)


def init_minusi(state, qubit):
    """
    Initialize qubit in state |-i>

    Args:
        state:
        qubit:

    Returns:

    """

    result = state.measure_collapse(qubit)

    if result == "1":
        state.x(qubit)

    H6(state, qubit)
