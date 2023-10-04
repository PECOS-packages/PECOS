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

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pecos.simulators.paulifaultprop.state import PauliFaultProp


def switch(state, switch_list: list[tuple[str, str]], qubit: int) -> None:
    for symbol_init, symbol_final in switch_list:
        if qubit in state.faults[symbol_init]:
            state.faults[symbol_init].remove(qubit)
            state.faults[symbol_final].add(qubit)
            break


def Identity(state: "PauliFaultProp", qubit: int, **params: Any) -> None:
    """Identity does nothing.

    X -> X
    Z -> Z
    Y -> Y

    Args:
    ----
        state (PauliFaultProp):  The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """


def X(state: "PauliFaultProp", qubit: int, **params: Any) -> None:
    """Pauli X.

    X -> X
    Z -> -Z
    Y -> -Y

    Args:
    ----
        state (PauliFaultProp):  The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """
    if state.track_sign and (qubit in state.faults["Z"] or qubit in state.faults["Y"]):
        state.flip_sign()


def Y(state: "PauliFaultProp", qubit: int, **params: Any) -> None:
    """X -> -X
    Z -> -Z
    Y -> Y.

    Args:
    ----
        state (PauliFaultProp):  The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """
    if state.track_sign and (qubit in state.faults["X"] or qubit in state.faults["Z"]):
        state.flip_sign()


def Z(state: "PauliFaultProp", qubit: int, **params: Any) -> None:
    """X -> -X
    Z -> Z
    Y -> -Y.

    Args:
    ----
        state (PauliFaultProp):  The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """
    if state.track_sign and (qubit in state.faults["X"] or qubit in state.faults["Y"]):
        state.flip_sign()


def SX(state: "PauliFaultProp", qubit: int, **params: Any) -> None:
    """Square root of X.

    X -> X
    Z -> -Y
    Y -> Z

    Args:
    ----
        state (PauliFaultProp):  The class representing the Pauli fault state.
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


def SXdg(state: "PauliFaultProp", qubit: int, **params: Any) -> None:
    """Hermitian conjugate of the square root of X.

    X -> X
    Z -> Y
    Y -> -Z

    Args:
    ----
        state (PauliFaultProp):  The class representing the Pauli fault state.
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


def SY(state: "PauliFaultProp", qubit: int, **params: Any) -> None:
    """Square root of Y.

    X -> -Z
    Z -> X
    Y -> Y

    Args:
    ----
        state (PauliFaultProp):  The class representing the Pauli fault state.
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


def SYdg(state: "PauliFaultProp", qubit: int, **params: Any) -> None:
    """Hermitian conjugate of the square root of Y.

    X -> Z
    Z -> -X
    Y -> Y

    Args:
    ----
        state (PauliFaultProp):  The class representing the Pauli fault state.
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


def SZ(state: "PauliFaultProp", qubit: int, **params: Any) -> None:
    """Square root of Z.

    X -> Y
    Z -> Z
    Y -> -X

    Args:
    ----
        state (PauliFaultProp):  The class representing the Pauli fault state.
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


def SZdg(state: "PauliFaultProp", qubit: int, **params: Any) -> None:
    """Hermitian conjugate of the square root of Z.

    X -> -Y
    Z -> Z
    Y -> X

    Args:
    ----
        state (PauliFaultProp):  The class representing the Pauli fault state.
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


def H(state: "PauliFaultProp", qubit: int, **params: Any) -> None:
    """Hadamard gate.

    X -> Z
    Z -> X
    Y -> -Y

    Args:
    ----
        state (PauliFaultProp):  The class representing the Pauli fault state.
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


def H2(state: "PauliFaultProp", qubit: int, **params: Any) -> None:
    """Hadamard-like rotation.

    X -> -Z
    Z -> -X
    Y -> -Y

    Args:
    ----
        state (PauliFaultProp):  The class representing the Pauli fault state.
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


def H3(state: "PauliFaultProp", qubit: int, **params: Any) -> None:
    """Hadamard-like rotation.

    X -> Y
    Z -> -Z
    Y -> X

    Args:
    ----
        state (PauliFaultProp):  The class representing the Pauli fault state.
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


def H4(state: "PauliFaultProp", qubit: int, **params: Any) -> None:
    """Hadamard-like rotation.

    X -> -Y
    Z -> -Z
    Y -> -X

    Args:
    ----
        state (PauliFaultProp):  The class representing the Pauli fault state.
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


def H5(state: "PauliFaultProp", qubit: int, **params: Any) -> None:
    """Hadamard-like rotation.

    X -> -X
    Z -> Y
    Y -> Z

    Args:
    ----
        state (PauliFaultProp):  The class representing the Pauli fault state.
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


def H6(state: "PauliFaultProp", qubit: int, **params: Any) -> None:
    """Hadamard-like rotation.

    X -> -X
    Z -> -Y
    Y -> -Z

    Args:
    ----
        state (PauliFaultProp):  The class representing the Pauli fault state.
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


def F(state: "PauliFaultProp", qubit: int, **params: Any) -> None:
    """Face rotation.

    X -> Y
    Z -> X
    Y -> Z

    Args:
    ----
        state (PauliFaultProp):  The class representing the Pauli fault state.
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


def F2(state: "PauliFaultProp", qubit: int, **params: Any) -> None:
    """Face rotation.

    X -> -Z
    Z -> Y
    Y -> -X

    Args:
    ----
        state (PauliFaultProp):  The class representing the Pauli fault state.
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


def F3(state: "PauliFaultProp", qubit: int, **params: Any) -> None:
    """Face rotation.

    X -> Y
    Z -> -X
    Y -> -Z

    Args:
    ----
        state (PauliFaultProp):  The class representing the Pauli fault state.
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


def F4(state: "PauliFaultProp", qubit: int, **params: Any) -> None:
    """Face rotation.

    X -> Z
    Z -> -Y
    Y -> -X

    Args:
    ----
        state (PauliFaultProp):  The class representing the Pauli fault state.
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


def Fdg(state: "PauliFaultProp", qubit: int, **params: Any) -> None:
    """Face rotation.

    X -> Z
    Z -> Y
    Y -> X

    Args:
    ----
        state (PauliFaultProp):  The class representing the Pauli fault state.
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


def F2dg(state: "PauliFaultProp", qubit: int, **params: Any) -> None:
    """Face rotation.

    X -> -Y
    Z -> -X
    Y -> Z

    Args:
    ----
        state (PauliFaultProp):  The class representing the Pauli fault state.
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


def F3dg(state: "PauliFaultProp", qubit: int, **params: Any) -> None:
    """Face rotation.

    X -> -Z
    Z -> -Y
    Y -> X

    Args:
    ----
        state (PauliFaultProp):  The class representing the Pauli fault state.
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


def F4dg(state: "PauliFaultProp", qubit: int, **params: Any) -> None:
    """Face rotation.

    X -> -Y
    Z -> X
    Y -> -Z

    Args:
    ----
        state (PauliFaultProp):  The class representing the Pauli fault state.
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
