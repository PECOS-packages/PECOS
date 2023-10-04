# Copyright 2018 The PECOS Developers
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

from typing import Any


def meas_x(state, qubit: int, **params: Any) -> int:
    """Measurement in the X basis.

    Args:
    ----
        state:
        qubit:
        **params:

    Returns:
    -------

    """
    if qubit in state.faults["Z"] or qubit in state.faults["Y"]:
        return 1
    else:
        return 0


def meas_z(state, qubit: int, **params: Any) -> int:
    """Measurement in the Z basis.

    Args:
    ----
        state:
        qubit:
        **params:

    Returns:
    -------

    """
    if qubit in state.faults["X"] or qubit in state.faults["Y"]:
        return 1
    else:
        return 0


def meas_y(state, qubit: int, **params: Any):
    """Measurement in the Y basis.

    Args:
    ----
        state:
        qubit:
        **params:

    Returns:
    -------

    """
    if qubit in state.faults["X"] or qubit in state.faults["Z"]:
        return 1
    else:
        return 0


def meas_pauli(state, qubits: int | tuple[int, ...], **params: Any) -> int:
    pauli = params["Pauli"]

    if isinstance(qubits, int) and pauli not in ["X", "Y", "Z"]:
        msg = "Pauli for a single qubit measurement must be 'X', 'Y' or 'Z'!"
        raise Exception(msg)

    if pauli in ["X", "Y", "Z"]:
        pauli = pauli * len(qubits)
    else:
        if len(pauli) == len(qubits) + 1:
            # last qubit is considered the syndrome ancilla
            qubits = qubits[:-1]
        elif len(pauli) != len(qubits):
            msg = "The Pauli operator needs to be the size of the qubits it is acting on or a single type."
            raise Exception(msg)

    meas = 0

    for q, p in zip(qubits, pauli):
        if p == "X":
            meas += meas_x(state, q)
        elif p == "Z":
            meas += meas_z(state, q)
        elif p == "Y":
            meas += meas_y(state, q)
        else:
            msg = "Pauli symbol not supported!"
            raise Exception(msg)

    return meas % 2


def force_output(state, qubit: int, forced_output: int = -1) -> int:
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
