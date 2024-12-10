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

from pecos.error_models.noise_impl_old.gate_groups import (
    error_one_paulis_collection,
    error_two_paulis_collection,
)

if TYPE_CHECKING:
    from pecos import QuantumCircuit


def noise_depolarizing_two_qubit_gates(
    locations: set[tuple[int, int]],
    after: QuantumCircuit,
    p: float,
) -> None:
    """Symmetric depolarizing noise for two-qubit gates.

    # TODO: Describe noise model

    Args:
    ----
        locations: Set of tuples of qubit pairs the ideal gates act on.
        after: QuantumCircuit collecting the noise that occurs after the ideal gates.
    """
    rand_nums = np.random.random(len(locations)) <= p

    for r, (loc1, loc2) in zip(rand_nums, locations):
        if r:
            index = np.random.choice(len(error_two_paulis_collection))
            err1, err2 = error_two_paulis_collection[index]

            if err1:
                after.append(err1, {loc1})

            if err2:
                after.append(err2, {loc2})


def noise_two_qubit_gates_depolarizing_with_noiseless(
    locations: set[tuple[int, int]],
    after: QuantumCircuit,
    p: float,
    noiseless_qubits: set[int] | None = None,
) -> None:
    """Noise for two-qubit gates.

    Args:
    ----
        locations: Set of tuples of qubit pairs the ideal gates act on.
        after: QuantumCircuit collecting the noise that occurs after the ideal gates.
    """
    rand_nums = np.random.random(len(locations)) <= p

    for r, (loc1, loc2) in zip(rand_nums, locations):
        if r:
            if loc1 in noiseless_qubits and loc1 in noiseless_qubits:
                continue

            elif loc1 in noiseless_qubits:
                err = np.random.choice(error_one_paulis_collection)
                after.append(err, {loc2})

            elif loc2 in noiseless_qubits:
                err = np.random.choice(error_one_paulis_collection)
                after.append(err, {loc1})

            else:
                index = np.random.choice(len(error_two_paulis_collection))
                err1, err2 = error_two_paulis_collection[index]

                if err1:
                    after.append(err1, {loc1})

                if err2:
                    after.append(err2, {loc2})
