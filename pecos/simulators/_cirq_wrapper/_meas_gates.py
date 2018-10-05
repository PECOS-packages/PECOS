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

from projectq.ops import Measure, H
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

    q = state.qids[qubit]

    Measure | q

    state.eng.flush()

    return int(q)


def meas_y(state, qubit):
    """
    Measurement in the Y-basis.

    Args:
        state:
        qubit:

    Returns:

    """

    q = state.qids[qubit]

    H5 | q

    meas_outcome = meas_z(state, qubit)

    H5 | q

    return meas_outcome


def meas_x(state, qubit):
    """
    Measurement in the X-basis.

    Args:
        state:
        qubit:

    Returns:

    """

    q = state.qids[qubit]

    H | q

    meas_outcome = meas_z(state, qubit)

    H | q

    return meas_outcome
