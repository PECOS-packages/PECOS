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

from typing import Any

from pecos.simulators.qulacs.gates_meas import meas_z
from pecos.simulators.qulacs.gates_one_qubit import X


def init_zero(state, qubit: int, **params: Any) -> None:
    """Initialise or reset the qubit to state |0>

    Args:
        state: An instance of Qulacs
        qubit: The index of the qubit to be initialised
    """
    result = meas_z(state, qubit)

    if result:
        X(state, qubit)


def init_one(state, qubit: int, **params: Any) -> None:
    """Initialise or reset the qubit to state |1>
    Args:
        state: An instance of Qulacs
        qubit: The index of the qubit to be initialised
    """
    result = meas_z(state, qubit)

    if not result:
        X(state, qubit)
