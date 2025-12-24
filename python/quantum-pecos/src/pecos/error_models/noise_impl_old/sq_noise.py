"""Legacy single-qubit noise implementation.

This module provides legacy noise models for single-qubit operations,
maintained for backward compatibility with existing error models
and simulations in PECOS.
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
from pecos.error_models.noise_impl_old.gate_groups import error_one_paulis_collection

if TYPE_CHECKING:
    from pecos import QuantumCircuit


def noise_depolarizing_sq_gate(
    locations: set[int],
    after: QuantumCircuit,
    p: float,
) -> None:
    """Apply a symmetric depolarizing noise model."""
    rand_nums = pc.random.random(len(locations)) <= p

    for r, loc in zip(rand_nums, locations, strict=False):
        if r:
            err = pc.random.choice(error_one_paulis_collection, 1)[0]
            after.append(err, {loc})
