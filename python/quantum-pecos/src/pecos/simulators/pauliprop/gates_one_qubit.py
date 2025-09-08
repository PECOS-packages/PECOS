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

"""Single-qubit gate operations for Pauli fault propagation simulator.

This module provides single-qubit quantum gate operations for the Pauli fault propagation simulator, including
Clifford gates and their effect on Pauli frame propagation for efficient stabilizer circuit simulation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos.simulators.pauliprop.state import PauliProp
    from pecos.typing import SimulatorGateParams


def switch(
    state: PauliProp,
    switch_list: list[tuple[str, str]],
    qubit: int,
) -> None:
    """Switch Pauli fault type on a qubit according to provided mapping.

    Transforms the fault type on a qubit by checking the current fault
    against a list of transformation rules and applying the first match.

    Args:
        state: Pauli fault propagation state containing fault sets.
        switch_list: List of (initial_fault, final_fault) transformation pairs.
        qubit: Index of the qubit to apply transformation to.
    """
    for symbol_init, symbol_final in switch_list:
        if qubit in state.faults[symbol_init]:
            state.faults[symbol_init].remove(qubit)
            state.faults[symbol_final].add(qubit)
            break


def Identity(state: PauliProp, qubit: int, **_params: SimulatorGateParams) -> None:
    """Identity does nothing.

    X -> X
    Z -> Z
    Y -> Y

    Args:
        state (PauliProp): The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """


def X(state: PauliProp, qubit: int, **_params: SimulatorGateParams) -> None:
    """Pauli X.

    X -> X
    Z -> -Z
    Y -> -Y

    Args:
        state (PauliProp): The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """
    if state.track_sign and (qubit in state.faults["Z"] or qubit in state.faults["Y"]):
        state.flip_sign()


def Y(state: PauliProp, qubit: int, **_params: SimulatorGateParams) -> None:
    """Apply Pauli Y gate.

    X -> -X
    Z -> -Z
    Y -> Y.

    Args:
        state (PauliProp): The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """
    if state.track_sign and (qubit in state.faults["X"] or qubit in state.faults["Z"]):
        state.flip_sign()


def Z(state: PauliProp, qubit: int, **_params: SimulatorGateParams) -> None:
    """Apply Pauli Z gate.

    X -> -X
    Z -> Z
    Y -> -Y.

    Args:
        state (PauliProp): The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """
    if state.track_sign and (qubit in state.faults["X"] or qubit in state.faults["Y"]):
        state.flip_sign()


def SX(state: PauliProp, qubit: int, **_params: SimulatorGateParams) -> None:
    """Square root of X.

    X -> X
    Z -> -Y
    Y -> Z

    Args:
        state (PauliProp): The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """
    if state.track_sign and qubit in state.faults["Z"]:
        state.flip_sign()

    switch(
        state,
        [
            ("Z", "Y"),
            ("Y", "Z"),
        ],
        qubit,
    )


def SXdg(state: PauliProp, qubit: int, **_params: SimulatorGateParams) -> None:
    """Hermitian conjugate of the square root of X.

    X -> X
    Z -> Y
    Y -> -Z

    Args:
        state (PauliProp): The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """
    if state.track_sign and qubit in state.faults["Y"]:
        state.flip_sign()

    switch(
        state,
        [
            ("Z", "Y"),
            ("Y", "Z"),
        ],
        qubit,
    )


def SY(state: PauliProp, qubit: int, **_params: SimulatorGateParams) -> None:
    """Square root of Y.

    X -> -Z
    Z -> X
    Y -> Y

    Args:
        state (PauliProp): The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """
    if state.track_sign and qubit in state.faults["X"]:
        state.flip_sign()

    switch(
        state,
        [
            ("X", "Z"),
            ("Z", "X"),
        ],
        qubit,
    )


def SYdg(state: PauliProp, qubit: int, **_params: SimulatorGateParams) -> None:
    """Hermitian conjugate of the square root of Y.

    X -> Z
    Z -> -X
    Y -> Y

    Args:
        state (PauliProp): The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """
    if state.track_sign and qubit in state.faults["Z"]:
        state.flip_sign()

    switch(
        state,
        [
            ("X", "Z"),
            ("Z", "X"),
        ],
        qubit,
    )


def SZ(state: PauliProp, qubit: int, **_params: SimulatorGateParams) -> None:
    """Square root of Z.

    X -> Y
    Z -> Z
    Y -> -X

    Args:
        state (PauliProp): The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """
    if state.track_sign and qubit in state.faults["Y"]:
        state.flip_sign()

    switch(
        state,
        [
            ("X", "Y"),
            ("Y", "X"),
        ],
        qubit,
    )


def SZdg(state: PauliProp, qubit: int, **_params: SimulatorGateParams) -> None:
    """Hermitian conjugate of the square root of Z.

    X -> -Y
    Z -> Z
    Y -> X

    Args:
        state (PauliProp): The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """
    if state.track_sign and qubit in state.faults["X"]:
        state.flip_sign()

    switch(
        state,
        [
            ("X", "Y"),
            ("Y", "X"),
        ],
        qubit,
    )


def H(state: PauliProp, qubit: int, **_params: SimulatorGateParams) -> None:
    """Hadamard gate.

    X -> Z
    Z -> X
    Y -> -Y

    Args:
        state (PauliProp): The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """
    if state.track_sign and qubit in state.faults["Y"]:
        state.flip_sign()

    switch(
        state,
        [
            ("X", "Z"),
            ("Z", "X"),
        ],
        qubit,
    )


def H2(state: PauliProp, qubit: int, **_params: SimulatorGateParams) -> None:
    """Hadamard-like rotation.

    X -> -Z
    Z -> -X
    Y -> -Y

    Args:
        state (PauliProp): The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """
    if state.track_sign:
        state.flip_sign()

    switch(
        state,
        [
            ("X", "Z"),
            ("Z", "X"),
        ],
        qubit,
    )


def H3(state: PauliProp, qubit: int, **_params: SimulatorGateParams) -> None:
    """Hadamard-like rotation.

    X -> Y
    Z -> -Z
    Y -> X

    Args:
        state (PauliProp): The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """
    if state.track_sign and qubit in state.faults["Z"]:
        state.flip_sign()

    switch(
        state,
        [
            ("X", "Y"),
            ("Y", "X"),
        ],
        qubit,
    )


def H4(state: PauliProp, qubit: int, **_params: SimulatorGateParams) -> None:
    """Hadamard-like rotation.

    X -> -Y
    Z -> -Z
    Y -> -X

    Args:
        state (PauliProp): The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """
    if state.track_sign:
        state.flip_sign()

    switch(
        state,
        [
            ("X", "Y"),
            ("Y", "X"),
        ],
        qubit,
    )


def H5(state: PauliProp, qubit: int, **_params: SimulatorGateParams) -> None:
    """Hadamard-like rotation.

    X -> -X
    Z -> Y
    Y -> Z

    Args:
        state (PauliProp): The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """
    if state.track_sign and qubit in state.faults["X"]:
        state.flip_sign()

    switch(
        state,
        [
            ("Z", "Y"),
            ("Y", "Z"),
        ],
        qubit,
    )


def H6(state: PauliProp, qubit: int, **_params: SimulatorGateParams) -> None:
    """Hadamard-like rotation.

    X -> -X
    Z -> -Y
    Y -> -Z

    Args:
        state (PauliProp): The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """
    if state.track_sign:
        state.flip_sign()

    switch(
        state,
        [
            ("Z", "Y"),
            ("Y", "Z"),
        ],
        qubit,
    )


def F(state: PauliProp, qubit: int, **_params: SimulatorGateParams) -> None:
    """Face rotation.

    X -> Y
    Z -> X
    Y -> Z

    Args:
        state (PauliProp): The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """
    switch(
        state,
        [
            ("X", "Y"),
            ("Z", "X"),
            ("Y", "Z"),
        ],
        qubit,
    )


def F2(state: PauliProp, qubit: int, **_params: SimulatorGateParams) -> None:
    """Face rotation.

    X -> -Z
    Z -> Y
    Y -> -X

    Args:
        state (PauliProp): The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """
    if state.track_sign and (qubit in state.faults["X"] or qubit in state.faults["Y"]):
        state.flip_sign()

    switch(
        state,
        [
            ("X", "Z"),
            ("Z", "Y"),
            ("Y", "X"),
        ],
        qubit,
    )


def F3(state: PauliProp, qubit: int, **_params: SimulatorGateParams) -> None:
    """Face rotation.

    X -> Y
    Z -> -X
    Y -> -Z

    Args:
        state (PauliProp): The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """
    if state.track_sign and (qubit in state.faults["Z"] or qubit in state.faults["Y"]):
        state.flip_sign()

    switch(
        state,
        [
            ("X", "Y"),
            ("Z", "X"),
            ("Y", "Z"),
        ],
        qubit,
    )


def F4(state: PauliProp, qubit: int, **_params: SimulatorGateParams) -> None:
    """Face rotation.

    X -> Z
    Z -> -Y
    Y -> -X

    Args:
        state (PauliProp): The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """
    if state.track_sign and (qubit in state.faults["Z"] or qubit in state.faults["Y"]):
        state.flip_sign()

    switch(
        state,
        [
            ("X", "Z"),
            ("Z", "Y"),
            ("Y", "X"),
        ],
        qubit,
    )


def Fdg(state: PauliProp, qubit: int, **_params: SimulatorGateParams) -> None:
    """Face rotation.

    X -> Z
    Z -> Y
    Y -> X

    Args:
        state (PauliProp): The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """
    switch(
        state,
        [
            ("X", "Z"),
            ("Z", "Y"),
            ("Y", "X"),
        ],
        qubit,
    )


def F2dg(state: PauliProp, qubit: int, **_params: SimulatorGateParams) -> None:
    """Face rotation.

    X -> -Y
    Z -> -X
    Y -> Z

    Args:
        state (PauliProp): The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """
    if state.track_sign and (qubit in state.faults["X"] or qubit in state.faults["Z"]):
        state.flip_sign()

    switch(
        state,
        [
            ("X", "Y"),
            ("Z", "X"),
            ("Y", "Z"),
        ],
        qubit,
    )


def F3dg(state: PauliProp, qubit: int, **_params: SimulatorGateParams) -> None:
    """Face rotation.

    X -> -Z
    Z -> -Y
    Y -> X

    Args:
        state (PauliProp): The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """
    if state.track_sign and (qubit in state.faults["X"] or qubit in state.faults["Z"]):
        state.flip_sign()

    switch(
        state,
        [
            ("X", "Z"),
            ("Z", "Y"),
            ("Y", "X"),
        ],
        qubit,
    )


def F4dg(state: PauliProp, qubit: int, **_params: SimulatorGateParams) -> None:
    """Face rotation.

    X -> -Y
    Z -> X
    Y -> -Z

    Args:
        state (PauliProp): The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """
    if state.track_sign and (qubit in state.faults["X"] or qubit in state.faults["Y"]):
        state.flip_sign()

    switch(
        state,
        [
            ("X", "Y"),
            ("Z", "X"),
            ("Y", "Z"),
        ],
        qubit,
    )
