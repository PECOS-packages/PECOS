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

"""Quantum measurement operations for Pauli fault propagation simulator.

This module provides quantum measurement operations for the Pauli fault propagation simulator, including projective
measurements with efficient Pauli frame updates and fault propagation tracking through measurement operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos.simulators.pauliprop.state import PauliProp
    from pecos.typing import SimulatorGateParams


def meas_x(state: PauliProp, qubit: int, **_params: SimulatorGateParams) -> int:
    """Measurement in the X basis.

    Args:
        state: The PauliProp state instance.
        qubit (int): The qubit index to measure.
        **_params: Unused additional parameters (kept for interface compatibility).
    """
    if qubit in state.faults["Z"] or qubit in state.faults["Y"]:
        return 1
    return 0


def meas_z(state: PauliProp, qubit: int, **_params: SimulatorGateParams) -> int:
    """Measurement in the Z basis.

    Args:
        state: The PauliProp state instance.
        qubit (int): The qubit index to measure.
        **_params: Unused additional parameters (kept for interface compatibility).
    """
    if qubit in state.faults["X"] or qubit in state.faults["Y"]:
        return 1
    return 0


def meas_y(state: PauliProp, qubit: int, **_params: SimulatorGateParams) -> int:
    """Measurement in the Y basis.

    Args:
        state: The PauliProp state instance.
        qubit (int): The qubit index to measure.
        **_params: Unused additional parameters (kept for interface compatibility).
    """
    if qubit in state.faults["X"] or qubit in state.faults["Z"]:
        return 1
    return 0


def meas_pauli(
    state: PauliProp,
    qubits: int | tuple[int, ...],
    **params: SimulatorGateParams,
) -> int:
    """Measure a multi-qubit Pauli operator.

    Performs measurement of a specified Pauli operator on one or more qubits,
    returning the parity of individual Pauli measurements on each qubit.

    Args:
        state: Pauli fault propagation state containing fault information.
        qubits: Qubit index or tuple of qubit indices to measure.
        **params: Gate parameters including 'Pauli' specifying the operator.

    Returns:
        Measurement result (0 or 1) as parity of individual measurements.

    Raises:
        Exception: If Pauli operator specification is invalid.
    """
    pauli = params["Pauli"]

    if isinstance(qubits, int) and pauli not in {"X", "Y", "Z"}:
        msg = "Pauli for a single qubit measurement must be 'X', 'Y' or 'Z'!"
        raise Exception(msg)

    if pauli in {"X", "Y", "Z"}:
        pauli *= len(qubits)
    elif len(pauli) == len(qubits) + 1:
        # last qubit is considered the syndrome ancilla
        qubits = qubits[:-1]
    elif len(pauli) != len(qubits):
        msg = "The Pauli operator needs to be the size of the qubits it is acting on or a single type."
        raise Exception(msg)

    meas = 0

    for q, p in zip(qubits, pauli, strict=False):
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


def force_output(_state: PauliProp, _qubit: int, forced_output: int = -1) -> int:
    """Outputs value.

    Used for error generators to generate outputs when replacing measurements.

    Args:
        _state: Unused state parameter (kept for interface compatibility).
        _qubit (int): Unused qubit parameter (kept for interface compatibility).
        forced_output (int): The value to output.
    """
    return forced_output
