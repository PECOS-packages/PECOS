"""Single-qubit Pauli gate implementations.

This module provides Pauli gate implementations (X, Y, Z) which form
the fundamental error operators in quantum error correction and are
essential building blocks for stabilizer codes.
"""

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

import pecos as pc
from pecos.qeclib.qubit.qgate_base import QGate


class X(QGate):
    """The Pauli X unitary."""

    matrix = pc.array(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=complex,
    )

    pauli_rules = (
        ("X", "+X"),
        ("Z", "-Z"),
    )


class Y(QGate):
    """The Pauli Y unitary."""

    matrix = pc.array(
        [
            [0, -1j],
            [1j, 0],
        ],
        dtype=complex,
    )

    pauli_rules = (
        ("X", "-X"),
        ("Z", "-Z"),
    )


class Z(QGate):
    """The Pauli Z unitary."""

    matrix = pc.array(
        [
            [1, 0],
            [0, -1],
        ],
        dtype=complex,
    )

    pauli_rules = (
        ("X", "-X"),
        ("Z", "+Z"),
    )
