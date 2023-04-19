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

from ._1q_gates import H5


def force_output(state, qubit, forced_output=-1):
    """
    Outputs value.

    Used for error generators to generate outputs when replacing measurements.

    Args:
        state:
        qubit:
        forced_output:

    Returns:

    """
    return forced_output


def meas_z(state, qubit):
    """
    Measurement in the Z-basis.

    Args:
        state:
        qubit:

    Returns:

    """

    m = state.measure_collapse(qubit)

    return int(m)


def meas_y(state, qubit):
    """
    Measurement in the Y-basis.

    Args:
        state:
        qubit:

    Returns:

    """

    H5(state, qubit)
    m = state.measure_collapse(qubit)
    H5(state, qubit)

    return int(m)


def meas_x(state, qubit):
    """
    Measurement in the X-basis.

    Args:
        state:
        qubit:

    Returns:

    """

    state.h(qubit)
    m = state.measure_collapse(qubit)
    state.h(qubit)

    return int(m)
