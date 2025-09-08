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

"""Single-qubit gate operations for Qulacs simulator.

This module provides single-qubit quantum gate operations for the Qulacs simulator, including Pauli gates,
rotation gates, Hadamard gates, and other fundamental single-qubit operations using the Rust backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos.simulators.qulacs import Qulacs
    from pecos.typing import SimulatorGateParams


def identity(state: Qulacs, qubit: int, **_params: SimulatorGateParams) -> None:
    """Identity gate.

    Args:
        state: An instance of Qulacs
        qubit: The index of the qubit where the gate is applied
    """
    # Identity gate does nothing


def X(state: Qulacs, qubit: int, **_params: SimulatorGateParams) -> None:
    """Pauli X gate.

    Args:
        state: An instance of Qulacs
        qubit: The index of the qubit where the gate is applied
    """
    state.qulacs_state.run_1q_gate("X", qubit)


def Y(state: Qulacs, qubit: int, **_params: SimulatorGateParams) -> None:
    """Pauli Y gate.

    Args:
        state: An instance of Qulacs
        qubit: The index of the qubit where the gate is applied
    """
    state.qulacs_state.run_1q_gate("Y", qubit)


def Z(state: Qulacs, qubit: int, **_params: SimulatorGateParams) -> None:
    """Pauli Z gate.

    Args:
        state: An instance of Qulacs
        qubit: The index of the qubit where the gate is applied
    """
    state.qulacs_state.run_1q_gate("Z", qubit)


def H(state: Qulacs, qubit: int, **_params: SimulatorGateParams) -> None:
    """Hadamard gate.

    Args:
        state: An instance of Qulacs
        qubit: The index of the qubit where the gate is applied
    """
    state.qulacs_state.run_1q_gate("H", qubit)


def SX(state: Qulacs, qubit: int, **_params: SimulatorGateParams) -> None:
    """Square root of X gate.

    Args:
        state: An instance of Qulacs
        qubit: The index of the qubit where the gate is applied
    """
    state.qulacs_state.run_1q_gate("SX", qubit)


def SXdg(state: Qulacs, qubit: int, **_params: SimulatorGateParams) -> None:
    """Dagger of square root of X gate.

    Args:
        state: An instance of Qulacs
        qubit: The index of the qubit where the gate is applied
    """
    state.qulacs_state.run_1q_gate("SXdg", qubit)


def SY(state: Qulacs, qubit: int, **_params: SimulatorGateParams) -> None:
    """Square root of Y gate.

    Args:
        state: An instance of Qulacs
        qubit: The index of the qubit where the gate is applied
    """
    state.qulacs_state.run_1q_gate("SY", qubit)


def SYdg(state: Qulacs, qubit: int, **_params: SimulatorGateParams) -> None:
    """Dagger of square root of Y gate.

    Args:
        state: An instance of Qulacs
        qubit: The index of the qubit where the gate is applied
    """
    state.qulacs_state.run_1q_gate("SYdg", qubit)


def SZ(state: Qulacs, qubit: int, **_params: SimulatorGateParams) -> None:
    """Square root of Z gate (S gate).

    Args:
        state: An instance of Qulacs
        qubit: The index of the qubit where the gate is applied
    """
    state.qulacs_state.run_1q_gate("SZ", qubit)


def SZdg(state: Qulacs, qubit: int, **_params: SimulatorGateParams) -> None:
    """Dagger of square root of Z gate (S† gate).

    Args:
        state: An instance of Qulacs
        qubit: The index of the qubit where the gate is applied
    """
    state.qulacs_state.run_1q_gate("SZdg", qubit)


def T(state: Qulacs, qubit: int, **_params: SimulatorGateParams) -> None:
    """T gate.

    Args:
        state: An instance of Qulacs
        qubit: The index of the qubit where the gate is applied
    """
    state.qulacs_state.run_1q_gate("T", qubit)


def Tdg(state: Qulacs, qubit: int, **_params: SimulatorGateParams) -> None:
    """T dagger gate.

    Args:
        state: An instance of Qulacs
        qubit: The index of the qubit where the gate is applied
    """
    state.qulacs_state.run_1q_gate("Tdg", qubit)


def RX(
    state: Qulacs,
    qubit: int,
    angles: tuple[float] | list[float] | None = None,
    **params: SimulatorGateParams,
) -> None:
    """Rotation around X axis.

    Args:
        state: An instance of Qulacs
        qubit: The index of the qubit where the gate is applied
        angles: A tuple or list containing a single rotation angle in radians
        **params: Additional parameters, can include 'angle' (float) or 'angles' (list)
    """
    # Extract angle from various possible sources for compatibility
    if angles is not None:
        # Standard interface: angles as positional parameter (Qulacs compatibility)
        if hasattr(angles, "__len__"):
            if len(angles) != 1:
                msg = "RX gate must be given 1 angle parameter."
                raise ValueError(msg)
            angle = angles[0]
        else:
            # Allow single float for convenience
            angle = angles
    elif "angle" in params:
        # Qulacs style: angle as keyword parameter
        angle = params["angle"]
    elif "angles" in params:
        # Angles from kwargs
        angles_param = params["angles"]
        if hasattr(angles_param, "__len__"):
            if len(angles_param) != 1:
                msg = "RX gate must be given 1 angle parameter."
                raise ValueError(msg)
            angle = angles_param[0]
        else:
            angle = angles_param
    else:
        msg = "RX gate requires an 'angle' or 'angles' parameter"
        raise TypeError(msg)

    state.qulacs_state.run_1q_gate("RX", qubit, {"angle": angle})


def RY(
    state: Qulacs,
    qubit: int,
    angles: tuple[float] | list[float] | None = None,
    **params: SimulatorGateParams,
) -> None:
    """Rotation around Y axis.

    Args:
        state: An instance of Qulacs
        qubit: The index of the qubit where the gate is applied
        angles: A tuple or list containing a single rotation angle in radians
        **params: Additional parameters, can include 'angle' (float) or 'angles' (list)
    """
    # Extract angle from various possible sources for compatibility
    if angles is not None:
        # Standard interface: angles as positional parameter (Qulacs compatibility)
        if hasattr(angles, "__len__"):
            if len(angles) != 1:
                msg = "RY gate must be given 1 angle parameter."
                raise ValueError(msg)
            angle = angles[0]
        else:
            # Allow single float for convenience
            angle = angles
    elif "angle" in params:
        # Qulacs style: angle as keyword parameter
        angle = params["angle"]
    elif "angles" in params:
        # Angles from kwargs
        angles_param = params["angles"]
        if hasattr(angles_param, "__len__"):
            if len(angles_param) != 1:
                msg = "RY gate must be given 1 angle parameter."
                raise ValueError(msg)
            angle = angles_param[0]
        else:
            angle = angles_param
    else:
        msg = "RY gate requires an 'angle' or 'angles' parameter"
        raise TypeError(msg)

    state.qulacs_state.run_1q_gate("RY", qubit, {"angle": angle})


def RZ(
    state: Qulacs,
    qubit: int,
    angles: tuple[float] | list[float] | None = None,
    **params: SimulatorGateParams,
) -> None:
    """Rotation around Z axis.

    Args:
        state: An instance of Qulacs
        qubit: The index of the qubit where the gate is applied
        angles: A tuple or list containing a single rotation angle in radians
        **params: Additional parameters, can include 'angle' (float) or 'angles' (list)
    """
    # Extract angle from various possible sources for compatibility
    if angles is not None:
        # Standard interface: angles as positional parameter (Qulacs compatibility)
        if hasattr(angles, "__len__"):
            if len(angles) != 1:
                msg = "RZ gate must be given 1 angle parameter."
                raise ValueError(msg)
            angle = angles[0]
        else:
            # Allow single float for convenience
            angle = angles
    elif "angle" in params:
        # Qulacs style: angle as keyword parameter
        angle = params["angle"]
    elif "angles" in params:
        # Angles from kwargs
        angles_param = params["angles"]
        if hasattr(angles_param, "__len__"):
            if len(angles_param) != 1:
                msg = "RZ gate must be given 1 angle parameter."
                raise ValueError(msg)
            angle = angles_param[0]
        else:
            angle = angles_param
    else:
        msg = "RZ gate requires an 'angle' or 'angles' parameter"
        raise TypeError(msg)

    state.qulacs_state.run_1q_gate("RZ", qubit, {"angle": angle})


def R1XY(
    state: Qulacs,
    qubit: int,
    angles: tuple[float] | list[float] | None = None,
    **params: SimulatorGateParams,
) -> None:
    """Single-qubit rotation with two angles (experimental).

    Args:
        state: An instance of Qulacs
        qubit: The index of the qubit where the gate is applied
        angles: A tuple or list of two rotation angles
        **params: Additional parameters, can include 'angles' (list of 2 floats)
    """
    # Extract angles from angles parameter or params
    if angles is not None:
        if hasattr(angles, "__len__"):
            if len(angles) < 2:
                msg = "R1XY gate must be given 2 angle parameters."
                raise ValueError(msg)
            angle_list = list(angles[:2])
        else:
            msg = "R1XY gate requires a list or tuple of 2 angles."
            raise ValueError(msg)
    elif "angles" in params:
        angles_param = params["angles"]
        if hasattr(angles_param, "__len__"):
            if len(angles_param) < 2:
                msg = "R1XY gate must be given 2 angle parameters."
                raise ValueError(msg)
            angle_list = list(angles_param[:2])
        else:
            msg = "R1XY gate requires a list or tuple of 2 angles."
            raise ValueError(msg)
    else:
        msg = "R1XY gate requires 'angles' parameter with 2 values."
        raise TypeError(msg)

    state.qulacs_state.run_1q_gate("R1XY", qubit, {"angles": angle_list})


# Additional gate aliases and implementations for compatibility


def F(state: Qulacs, qubit: int, **_params: SimulatorGateParams) -> None:
    """F gate (F1 gate - qutrit Hadamard projected to 2 levels)."""
    # F gate has matrix [[1+i, 1-i], [1+i, -1+i]]/2
    # It's different from SX
    state.qulacs_state.run_1q_gate("F", qubit)


def Fdg(state: Qulacs, qubit: int, **_params: SimulatorGateParams) -> None:
    """F dagger gate."""
    state.qulacs_state.run_1q_gate("Fdg", qubit)


# Hadamard variants - these would need specific implementations
# For now, defaulting to standard Hadamard


def H2(state: Qulacs, qubit: int, **_params: SimulatorGateParams) -> None:
    """H2 gate variant."""
    state.qulacs_state.run_1q_gate("H2", qubit, {})


def H3(state: Qulacs, qubit: int, **_params: SimulatorGateParams) -> None:
    """H3 gate variant."""
    state.qulacs_state.run_1q_gate("H3", qubit, {})


def H4(state: Qulacs, qubit: int, **_params: SimulatorGateParams) -> None:
    """H4 gate variant."""
    state.qulacs_state.run_1q_gate("H4", qubit, {})


def H5(state: Qulacs, qubit: int, **_params: SimulatorGateParams) -> None:
    """H5 gate variant."""
    state.qulacs_state.run_1q_gate("H5", qubit, {})


def H6(state: Qulacs, qubit: int, **_params: SimulatorGateParams) -> None:
    """H6 gate variant."""
    state.qulacs_state.run_1q_gate("H6", qubit, {})


# F gate variants - similar to Hadamard variants


def F2(state: Qulacs, qubit: int, **_params: SimulatorGateParams) -> None:
    """F2 gate variant."""
    state.qulacs_state.run_1q_gate("F2", qubit, {})


def F2d(state: Qulacs, qubit: int, **_params: SimulatorGateParams) -> None:
    """F2 dagger gate variant."""
    state.qulacs_state.run_1q_gate("F2dg", qubit, {})


def F3(state: Qulacs, qubit: int, **_params: SimulatorGateParams) -> None:
    """F3 gate variant."""
    state.qulacs_state.run_1q_gate("F3", qubit, {})


def F3d(state: Qulacs, qubit: int, **_params: SimulatorGateParams) -> None:
    """F3 dagger gate variant."""
    state.qulacs_state.run_1q_gate("F3dg", qubit, {})


def F4(state: Qulacs, qubit: int, **_params: SimulatorGateParams) -> None:
    """F4 gate variant."""
    state.qulacs_state.run_1q_gate("F4", qubit, {})


def F4d(state: Qulacs, qubit: int, **_params: SimulatorGateParams) -> None:
    """F4 dagger gate variant."""
    state.qulacs_state.run_1q_gate("F4dg", qubit, {})
