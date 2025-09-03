# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Two-qubit gate operations for Qulacs simulator.

This module provides two-qubit quantum gate operations for the Qulacs simulator, including CNOT, CZ, SWAP,
and other two-qubit operations using the Rust backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos.simulators.qulacs import Qulacs
    from pecos.typing import SimulatorGateParams


def CX(
    state: Qulacs,
    control: int | tuple[int, int] | list[int],
    target: int | None = None,
    **_params: SimulatorGateParams,
) -> None:
    """CNOT gate (controlled X gate).

    Args:
        state: An instance of Qulacs
        control: Control qubit index, or tuple/list of (control, target)
        target: Target qubit index (if control is just an int)
    """
    # Handle both calling conventions
    if target is None:
        # Called with tuple/list: CX(state, (control, target))
        if isinstance(control, tuple | list):
            qubits = tuple(control)
        else:
            msg = "CX requires two qubits"
            raise ValueError(msg)
    else:
        # Called with separate args: CX(state, control, target)
        qubits = (control, target)

    state.qulacs_state.run_2q_gate("CX", qubits, None)


def CY(
    state: Qulacs,
    control: int | tuple[int, int] | list[int],
    target: int | None = None,
    **_params: SimulatorGateParams,
) -> None:
    """Controlled Y gate.

    Args:
        state: An instance of Qulacs
        control: Control qubit index, or tuple/list of (control, target)
        target: Target qubit index (if control is just an int)
    """
    # Handle both calling conventions
    if target is None:
        # Called with tuple/list: CY(state, (control, target))
        if isinstance(control, tuple | list):
            qubits = tuple(control)
        else:
            msg = "CY requires two qubits"
            raise ValueError(msg)
    else:
        # Called with separate args: CY(state, control, target)
        qubits = (control, target)

    state.qulacs_state.run_2q_gate("CY", qubits, None)


def CZ(
    state: Qulacs,
    control: int | tuple[int, int] | list[int],
    target: int | None = None,
    **_params: SimulatorGateParams,
) -> None:
    """Controlled Z gate.

    Args:
        state: An instance of Qulacs
        control: Control qubit index, or tuple/list of (control, target)
        target: Target qubit index (if control is just an int)
    """
    # Handle both calling conventions
    if target is None:
        # Called with tuple/list: CZ(state, (control, target))
        if isinstance(control, tuple | list):
            qubits = tuple(control)
        else:
            msg = "CZ requires two qubits"
            raise ValueError(msg)
    else:
        # Called with separate args: CZ(state, control, target)
        qubits = (control, target)

    state.qulacs_state.run_2q_gate("CZ", qubits, None)


def SWAP(
    state: Qulacs,
    qubit1: int | tuple[int, int] | list[int],
    qubit2: int | None = None,
    **_params: SimulatorGateParams,
) -> None:
    """SWAP gate.

    Args:
        state: An instance of Qulacs
        qubit1: First qubit index, or tuple/list of both qubits
        qubit2: Second qubit index (if qubit1 is just an int)
    """
    # Handle both calling conventions
    if qubit2 is None:
        # Called with tuple/list: SWAP(state, (qubit1, qubit2))
        if isinstance(qubit1, tuple | list):
            qubits = tuple(qubit1)
        else:
            msg = "SWAP requires two qubits"
            raise ValueError(msg)
    else:
        # Called with separate args: SWAP(state, qubit1, qubit2)
        qubits = (qubit1, qubit2)

    state.qulacs_state.run_2q_gate("SWAP", qubits, None)


def RXX(
    state: Qulacs,
    qubit1: int | tuple[int, int] | list[int],
    qubit2: int | None = None,
    angles: list[float] | None = None,
    **params: SimulatorGateParams,
) -> None:
    """RXX gate (two-qubit X rotation).

    Args:
        state: An instance of Qulacs
        qubit1: First qubit index, or tuple/list of both qubits
        qubit2: Second qubit index (if qubit1 is just an int)
        angles: List containing a single rotation angle in radians
        **params: Additional parameters, can include 'angle' (float) or 'angles' (list)
    """
    # Handle both calling conventions
    if qubit2 is None:
        # Called with tuple/list: RXX(state, (qubit1, qubit2))
        if isinstance(qubit1, tuple | list):
            qubits = tuple(qubit1)
        else:
            msg = "RXX requires two qubits"
            raise ValueError(msg)
    else:
        # Called with separate args: RXX(state, qubit1, qubit2)
        qubits = (qubit1, qubit2)

    # Extract angle from angles parameter or params
    if angles is not None and len(angles) > 0:
        angle = angles[0]
    elif "angles" in params and len(params["angles"]) > 0:
        angle = params["angles"][0]
    else:
        angle = 0.0

    state.qulacs_state.run_2q_gate("RXX", qubits, {"angle": angle})


def RYY(
    state: Qulacs,
    qubit1: int | tuple[int, int] | list[int],
    qubit2: int | None = None,
    angles: list[float] | None = None,
    **params: SimulatorGateParams,
) -> None:
    """RYY gate (two-qubit Y rotation).

    Args:
        state: An instance of Qulacs
        qubit1: First qubit index, or tuple/list of both qubits
        qubit2: Second qubit index (if qubit1 is just an int)
        angles: List containing a single rotation angle in radians
        **params: Additional parameters, can include 'angle' (float) or 'angles' (list)
    """
    # Handle both calling conventions
    if qubit2 is None:
        # Called with tuple/list: RYY(state, (qubit1, qubit2))
        if isinstance(qubit1, tuple | list):
            qubits = tuple(qubit1)
        else:
            msg = "RYY requires two qubits"
            raise ValueError(msg)
    else:
        # Called with separate args: RYY(state, qubit1, qubit2)
        qubits = (qubit1, qubit2)

    # Extract angle from angles parameter or params
    if angles is not None and len(angles) > 0:
        angle = angles[0]
    elif "angles" in params and len(params["angles"]) > 0:
        angle = params["angles"][0]
    else:
        angle = 0.0

    state.qulacs_state.run_2q_gate("RYY", qubits, {"angle": angle})


def RZZ(
    state: Qulacs,
    qubit1: int | tuple[int, int] | list[int],
    qubit2: int | None = None,
    angles: list[float] | None = None,
    **params: SimulatorGateParams,
) -> None:
    """RZZ gate (two-qubit Z rotation).

    Args:
        state: An instance of Qulacs
        qubit1: First qubit index, or tuple/list of both qubits
        qubit2: Second qubit index (if qubit1 is just an int)
        angles: List containing a single rotation angle in radians
        **params: Additional parameters, can include 'angle' (float) or 'angles' (list)
    """
    # Handle both calling conventions
    if qubit2 is None:
        # Called with tuple/list: RZZ(state, (qubit1, qubit2))
        if isinstance(qubit1, tuple | list):
            qubits = tuple(qubit1)
        else:
            msg = "RZZ requires two qubits"
            raise ValueError(msg)
    else:
        # Called with separate args: RZZ(state, qubit1, qubit2)
        qubits = (qubit1, qubit2)

    # Extract angle from angles parameter or params
    if angles is not None and len(angles) > 0:
        angle = angles[0]
    elif "angles" in params and len(params["angles"]) > 0:
        angle = params["angles"][0]
    else:
        angle = 0.0

    state.qulacs_state.run_2q_gate("RZZ", qubits, {"angle": angle})


def R2XXYYZZ(
    state: Qulacs,
    qubit1: int | tuple[int, int] | list[int],
    qubit2: int | None = None,
    angles: list[float] | None = None,
    **params: SimulatorGateParams,
) -> None:
    """Combined RXX, RYY, RZZ rotation gate.

    Args:
        state: An instance of Qulacs
        qubit1: First qubit index, or tuple/list of both qubits
        qubit2: Second qubit index (if qubit1 is just an int)
        angles: List of three angles for ZZ, YY, XX rotations (in that order)
        **params: Additional parameters, can include 'angles' (list of 3 floats)
    """
    # Handle both calling conventions
    if qubit2 is None:
        # Called with tuple/list: R2XXYYZZ(state, (qubit1, qubit2))
        if isinstance(qubit1, tuple | list):
            qubits = tuple(qubit1)
        else:
            msg = "R2XXYYZZ requires two qubits"
            raise ValueError(msg)
    else:
        # Called with separate args: R2XXYYZZ(state, qubit1, qubit2)
        qubits = (qubit1, qubit2)

    # Extract angles from angles parameter or params
    if angles is not None and len(angles) >= 3:
        angle_list = angles[:3]
    elif "angles" in params and len(params["angles"]) >= 3:
        angle_list = params["angles"][:3]
    else:
        angle_list = [0.0, 0.0, 0.0]

    # Apply RZZ, RYY, RXX in order (note the order matches RZZRYYRXX)
    state.qulacs_state.run_2q_gate("RZZRYYRXX", qubits, {"angles": angle_list})


def SXX(
    state: Qulacs,
    qubit1: int | tuple[int, int] | list[int],
    qubit2: int | None = None,
    **_params: SimulatorGateParams,
) -> None:
    """SXX gate (square root of XX).

    Args:
        state: An instance of Qulacs
        qubit1: First qubit index, or tuple/list of both qubits
        qubit2: Second qubit index (if qubit1 is just an int)
    """
    # Handle both calling conventions
    if qubit2 is None:
        # Called with tuple/list: SXX(state, (qubit1, qubit2))
        if isinstance(qubit1, tuple | list):
            qubits = tuple(qubit1)
        else:
            msg = "SXX requires two qubits"
            raise ValueError(msg)
    else:
        # Called with separate args: SXX(state, qubit1, qubit2)
        qubits = (qubit1, qubit2)

    state.qulacs_state.run_2q_gate("SXX", qubits, None)


def SXXdg(
    state: Qulacs,
    qubit1: int | tuple[int, int] | list[int],
    qubit2: int | None = None,
    **_params: SimulatorGateParams,
) -> None:
    """SXX dagger gate.

    Args:
        state: An instance of Qulacs
        qubit1: First qubit index, or tuple/list of both qubits
        qubit2: Second qubit index (if qubit1 is just an int)
    """
    # Handle both calling conventions
    if qubit2 is None:
        # Called with tuple/list: SXXdg(state, (qubit1, qubit2))
        if isinstance(qubit1, tuple | list):
            qubits = tuple(qubit1)
        else:
            msg = "SXXdg requires two qubits"
            raise ValueError(msg)
    else:
        # Called with separate args: SXXdg(state, qubit1, qubit2)
        qubits = (qubit1, qubit2)

    state.qulacs_state.run_2q_gate("SXXdg", qubits, None)


def SYY(
    state: Qulacs,
    qubit1: int | tuple[int, int] | list[int],
    qubit2: int | None = None,
    **_params: SimulatorGateParams,
) -> None:
    """SYY gate (square root of YY).

    Args:
        state: An instance of Qulacs
        qubit1: First qubit index, or tuple/list of both qubits
        qubit2: Second qubit index (if qubit1 is just an int)
    """
    # Handle both calling conventions
    if qubit2 is None:
        # Called with tuple/list: SYY(state, (qubit1, qubit2))
        if isinstance(qubit1, tuple | list):
            qubits = tuple(qubit1)
        else:
            msg = "SYY requires two qubits"
            raise ValueError(msg)
    else:
        # Called with separate args: SYY(state, qubit1, qubit2)
        qubits = (qubit1, qubit2)

    state.qulacs_state.run_2q_gate("SYY", qubits, None)


def SYYdg(
    state: Qulacs,
    qubit1: int | tuple[int, int] | list[int],
    qubit2: int | None = None,
    **_params: SimulatorGateParams,
) -> None:
    """SYY dagger gate.

    Args:
        state: An instance of Qulacs
        qubit1: First qubit index, or tuple/list of both qubits
        qubit2: Second qubit index (if qubit1 is just an int)
    """
    # Handle both calling conventions
    if qubit2 is None:
        # Called with tuple/list: SYYdg(state, (qubit1, qubit2))
        if isinstance(qubit1, tuple | list):
            qubits = tuple(qubit1)
        else:
            msg = "SYYdg requires two qubits"
            raise ValueError(msg)
    else:
        # Called with separate args: SYYdg(state, qubit1, qubit2)
        qubits = (qubit1, qubit2)

    state.qulacs_state.run_2q_gate("SYYdg", qubits, None)


def SZZ(
    state: Qulacs,
    qubit1: int | tuple[int, int] | list[int],
    qubit2: int | None = None,
    **_params: SimulatorGateParams,
) -> None:
    """SZZ gate (square root of ZZ).

    Args:
        state: An instance of Qulacs
        qubit1: First qubit index, or tuple/list of both qubits
        qubit2: Second qubit index (if qubit1 is just an int)
    """
    # Handle both calling conventions
    if qubit2 is None:
        # Called with tuple/list: SZZ(state, (qubit1, qubit2))
        if isinstance(qubit1, tuple | list):
            qubits = tuple(qubit1)
        else:
            msg = "SZZ requires two qubits"
            raise ValueError(msg)
    else:
        # Called with separate args: SZZ(state, qubit1, qubit2)
        qubits = (qubit1, qubit2)

    state.qulacs_state.run_2q_gate("SZZ", qubits, None)


def SZZdg(
    state: Qulacs,
    qubit1: int | tuple[int, int] | list[int],
    qubit2: int | None = None,
    **_params: SimulatorGateParams,
) -> None:
    """SZZ dagger gate.

    Args:
        state: An instance of Qulacs
        qubit1: First qubit index, or tuple/list of both qubits
        qubit2: Second qubit index (if qubit1 is just an int)
    """
    # Handle both calling conventions
    if qubit2 is None:
        # Called with tuple/list: SZZdg(state, (qubit1, qubit2))
        if isinstance(qubit1, tuple | list):
            qubits = tuple(qubit1)
        else:
            msg = "SZZdg requires two qubits"
            raise ValueError(msg)
    else:
        # Called with separate args: SZZdg(state, qubit1, qubit2)
        qubits = (qubit1, qubit2)

    state.qulacs_state.run_2q_gate("SZZdg", qubits, None)


def G(
    state: Qulacs,
    qubit1: int | tuple[int, int] | list[int],
    qubit2: int | None = None,
    **_params: SimulatorGateParams,
) -> None:
    """G gate (special two-qubit gate).

    Args:
        state: An instance of Qulacs
        qubit1: First qubit index, or tuple/list of both qubits
        qubit2: Second qubit index (if qubit1 is just an int)
    """
    # Handle both calling conventions
    if qubit2 is None:
        # Called with tuple/list: G(state, (qubit1, qubit2))
        if isinstance(qubit1, tuple | list):
            qubits = tuple(qubit1)
        else:
            msg = "G requires two qubits"
            raise ValueError(msg)
    else:
        # Called with separate args: G(state, qubit1, qubit2)
        qubits = (qubit1, qubit2)

    state.qulacs_state.run_2q_gate("G2", qubits, None)
