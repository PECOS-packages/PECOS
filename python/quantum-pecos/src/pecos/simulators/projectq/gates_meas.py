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

from projectq.ops import Measure

from pecos.simulators.projectq.gates_one_qubit import H5, H


def force_output(state, qubit, forced_output=-1, **params: Any):
    """Outputs value.

    Used for error generators to generate outputs when replacing measurements.

    Args:
    ----
        state:
        qubit:
        forced_output:

    Returns:
    -------

    """
    return forced_output


def meas_z(state, qubit, forced_outcome=-1, **params: Any):
    """Measurement in the Z-basis.

    Args:
        state:
        qubit:
        forced_outcome:
        **params:

    Returns:

    """
    q = state.qids[qubit]

    state.eng.flush()

    if forced_outcome in {0, 1}:
        # project the qubit to the desired state ("randomly" chooses the value `forced_outcome`)
        state.eng.backend.collapse_wavefunction([q], [forced_outcome])
        # Note: this will raise an error if the probability of collapsing to this state is close to 0.0

        return forced_outcome

    else:
        Measure | q
        state.eng.flush()

        return int(q)


def meas_y(state, qubit, forced_outcome=-1, **params: Any):
    """Measurement in the Y-basis.

    Args:
    ----
        state:
        qubit:
        forced_outcome:

    Returns:
    -------

    """
    H5(state, qubit)
    meas_outcome = meas_z(state, qubit, forced_outcome)
    H5(state, qubit)

    return meas_outcome


def meas_x(state, qubit, forced_outcome=-1, **params: Any):
    """Measurement in the X-basis.

    Args:
    ----
        state:
        qubit:
        forced_outcome:

    Returns:
    -------

    """
    H(state, qubit)
    meas_outcome = meas_z(state, qubit, forced_outcome)
    H(state, qubit)

    return meas_outcome
