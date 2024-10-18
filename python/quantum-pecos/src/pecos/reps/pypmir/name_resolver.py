# Copyright 2024 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import numpy as np

from pecos.reps.pypmir.op_types import QOp
from pecos.tools.find_cliffs import r1xy2cliff, rz2cliff


def sim_name_resolver(qop: QOp) -> str:
    """Takes the name of the operation and translates it to something all the simulators recognize."""

    # TODO: Support conversion of all SQ gates.
    # TODO: Try to support as many TQ gates... but at least all the allowed ones to standard TQ Cliffords

    if qop.name == "RZZ" and qop.angles == (0.0,):
        return "I"

    elif qop.name == "RZZ":
        (theta,) = qop.angles
        theta = theta % (2 * np.pi)

        if np.isclose(theta, np.pi / 2, atol=1e-12):
            return "SZZ"
        elif np.isclose(theta, np.pi * (3 / 2), atol=1e-12):
            return "SZZdg"

    elif qop.name == "RZ":
        (angle,) = qop.angles
        sym = rz2cliff(angle)
        if isinstance(sym, str):
            return sym

    elif qop.name == "R1XY":
        angles = qop.angles
        sym = r1xy2cliff(*angles)
        if isinstance(sym, str):
            return sym

    return qop.name
