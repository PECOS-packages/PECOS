# Copyright 2021 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pecos import QuantumCircuit


def noise_meas_bitflip(
    locations: set[int],
    metadata: dict,
    after: QuantumCircuit,
    p: float,
) -> None:
    """Bit-flip noise model for measurements.

    Args:
    ----
        locations: Set of qubits the ideal gates act on.
        metadata: Extra information about the gate.
        after: QuantumCircuit collecting the noise that occurs after the ideal gates.
        flip: The symbol for what Pauli operator should be applied if a measurement fault occurs.
    """
    # Bit flip noise
    # --------------
    rand_nums = np.random.random(len(locations)) <= p

    for r, loc in zip(rand_nums, locations):
        if r:
            var = (
                metadata["var_output"][loc]
                if metadata.get("var_output")
                else metadata["var"]
            )

            after.append(
                "cop",
                {loc},
                expr={"t": var, "a": var, "op": "^", "b": 1},
            )  # flip output bit
