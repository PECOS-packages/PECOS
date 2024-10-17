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

import numpy as np


def meas_z(state, qubit: int, **params: Any) -> int:
    """Measure in the Z-basis, collapse and normalise.

    Notes:
        The number of qubits in the state remains the same.

    Args:
        state: An instance of BasicSV
        qubit: The index of the qubit to be measured

    Returns:
        The outcome of the measurement, either 0 or 1.
    """
    if qubit >= state.num_qubits or qubit < 0:
        msg = f"Qubit {qubit} out of range."
        raise ValueError(msg)

    # Define the Z-basis projectors
    proj_0 = np.array(
        [
            [1, 0],
            [0, 0],
        ],
    )
    proj_1 = np.array(
        [
            [0, 0],
            [0, 1],
        ],
    )

    # Use np.einsum to apply the projector to `qubit`.
    # To do so, we need to assign subscript labels to each array axis.
    subscripts = "".join(
        [
            state.subscript_string((qubit,), ("q",)),  # Current vector
            ",",
            "Qq",  # Subscripts for the projector operator, acting on `qubit` q
            "->",
            state.subscript_string(
                (qubit,),
                ("Q",),
            ),  # Resulting vector, with updated Q
        ],
    )
    # Obtain the projected state where qubit `qubit` is in state |0>
    projected_vector = np.einsum(subscripts, state.internal_vector, proj_0)

    # Obtain the probability of outcome |0>
    prob_0 = np.sum(projected_vector * np.conjugate(projected_vector))

    # Simulate the measurement
    if np.random.random() < prob_0:
        result = 0
    else:
        result = 1

    # Collapse the state
    if result == 0:
        state.internal_vector = projected_vector
    else:
        state.internal_vector = np.einsum(subscripts, state.internal_vector, proj_1)

    # Normalise
    if result == 0:
        state.internal_vector = state.internal_vector / np.sqrt(prob_0)
    else:
        state.internal_vector = state.internal_vector / np.sqrt(1 - prob_0)

    return result
