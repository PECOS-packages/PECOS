"""Legacy initialization noise implementation.

This module provides legacy noise models for qubit initialization
operations, maintained for backward compatibility with existing
error models and simulations in PECOS.
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
    from collections.abc import Sequence

    from pecos import QuantumCircuit


def noise_init_bitflip(
    locations: Sequence[int],
    after: QuantumCircuit,
    flip: str,
    p: float,
) -> None:
    """The noise model for qubit (re)initialization.

    Args:
    ----
        locations: Set of qubits the ideal gates act on.
        after: QuantumCircuit collecting the noise that occurs after the ideal gates.
        flip: The symbol for what Pauli operator should be applied if an initialization fault occurs.
        p: The probability of a bit-flip error occurring during initialization.
    """
    rand_nums = pc.random.random(len(locations)) <= p

    for r, loc in zip(rand_nums, locations, strict=False):
        if r:
            after.append(flip, {loc})
