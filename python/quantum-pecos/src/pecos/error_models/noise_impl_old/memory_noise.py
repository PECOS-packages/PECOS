"""Legacy memory noise implementation.

This module provides legacy noise models for quantum memory operations
and idle qubit errors, maintained for backward compatibility with
existing error models and simulations in PECOS.
"""

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

import pecos as pc

if TYPE_CHECKING:
    from pecos import QuantumCircuit


def noise_tq_mem(
    locations: set[tuple[int, int]],
    after: QuantumCircuit,
    p: float,
) -> None:
    """The memory noise model for idling qubits.

    Args:
    ----
        locations: Set of qubits the ideal gates act on.
        after: QuantumCircuit collecting the noise that occurs after the ideal gates.
        p: The probability of a Z error occurring on each idling qubit.
    """
    err_qubits = set()
    for locs in locations:
        rand_nums = pc.random.random(len(locs)) <= p

        for r, loc in zip(rand_nums, locs, strict=False):
            if r:
                err_qubits.add(loc)

    if err_qubits:
        after.append("Z", err_qubits)
