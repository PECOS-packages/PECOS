# -*- coding: utf-8 -*-

#  =========================================================================  #
#   Copyright 2018 Ciarán Ryan-Anderson
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#  =========================================================================  #

from typing import List, Tuple
from .state import PauliFaultProp


def switch(state,
           switch_list: List[Tuple[str, str]],
           qubit: int) -> None:

    for symbol_init, symbol_final in switch_list:
        if qubit in state.faults[symbol_init]:
            state.faults[symbol_init].remove(qubit)
            state.faults[symbol_final].add(qubit)
            break


def I(state: PauliFaultProp,
      qubit: int) -> None:
    """
    Identity does nothing.

    X -> X

    Z -> Z

    Y -> Y

    Args:
        state (PauliFaultProp):  The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """
    pass


def X(state,
      qubit: int) -> None:
    """
    Pauli X

    X -> X

    Z -> -Z

    Y -> -Y

    Args:
        state (PauliFaultProp):  The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """
    pass


def Y(state,
      qubit: int) -> None:
    """
    X -> -X

    Z -> -Z

    Y -> Y

    Args:
        state (PauliFaultProp):  The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """
    pass


def Z(state,
      qubit: int) -> None:
    """
    X -> -X

    Z -> Z

    Y -> -Y

    Args:
        state (PauliFaultProp):  The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """
    pass


def Q(state,
      qubit: int) -> None:
    """
    Square root of X.

    X -> X

    Z -> -Y

    Y -> Z

    Args:
        state (PauliFaultProp):  The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """

    switch(state, [
        ('Z', 'Y'),
        ('Y', 'Z')
    ], qubit)


def Qd(state,
       qubit: int) -> None:
    """
    Hermitian conjugate of the square root of X.

    X -> X

    Z -> Y

    Y -> -Z

    Args:
        state (PauliFaultProp):  The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """
    switch(state, [
        ('Z', 'Y'),
        ('Y', 'Z')
    ], qubit)


def R(state,
      qubit: int) -> None:
    """
    Square root of Y.

    X -> -Z

    Z -> X

    Y -> Y

    Args:
        state (PauliFaultProp):  The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """
    switch(state, [
        ('X', 'Z'),
        ('Z', 'X')
    ], qubit)


def Rd(state,
       qubit: int) -> None:
    """
    Hermitian conjugate of the square root of Y.

    X -> Z

    Z -> -X

    Y -> Y

    Args:
        state (PauliFaultProp):  The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """

    switch(state, [
        ('X', 'Z'),
        ('Z', 'X')
    ], qubit)


def S(state,
      qubit: int) -> None:
    """
    Square root of Z.

    X -> Y

    Z -> Z

    Y -> -X

    Args:
        state (PauliFaultProp):  The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """

    switch(state, [
        ('X', 'Y'),
        ('Y', 'X')
    ], qubit)


def Sd(state,
       qubit: int) -> None:
    """
    Square root of Z.

    X -> -Y

    Z -> Z

    Y -> X

    Args:
        state (PauliFaultProp):  The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """

    switch(state, [
        ('X', 'Y'),
        ('Y', 'X')
    ], qubit)


def H(state,
      qubit: int) -> None:
    """
    Square root of Z.

    X -> Z

    Z -> X

    Y -> -Y

    Args:
        state (PauliFaultProp):  The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """

    switch(state, [
        ('X', 'Z'),
        ('Z', 'X')
    ], qubit)


def H2(state,
       qubit: int) -> None:
    """
    Square root of Z.

    X -> -Z

    Z -> -X

    Y -> -Y

    Args:
        state (PauliFaultProp):  The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """

    switch(state, [
        ('X', 'Z'),
        ('Z', 'X')
    ], qubit)


def H3(state,
       qubit: int) -> None:
    """
    Square root of Z.

    X -> Y
    Z -> -Z
    Y -> X

    Args:
        state (PauliFaultProp):  The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """

    switch(state, [
        ('X', 'Y'),
        ('Y', 'X')
    ], qubit)


def H4(state,
       qubit: int) -> None:
    """
    Square root of Z.

    X -> -Y
    Z -> -Z
    Y -> -X

    Args:
        state (PauliFaultProp):  The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """

    switch(state, [
        ('X', 'Y'),
        ('Y', 'X')
    ], qubit)


def H5(state,
       qubit: int) -> None:
    """
    Square root of Z.

    X -> -X
    Z -> Y
    Y -> Z

    Args:
        state (PauliFaultProp):  The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """

    switch(state, [
        ('Z', 'Y'),
        ('Y', 'Z')
    ], qubit)


def H6(state,
       qubit: int) -> None:
    """
    Square root of Z.

    X -> -X
    Z -> -Y
    Y -> -Z

    Args:
        state (PauliFaultProp):  The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """

    switch(state, [
        ('Z', 'Y'),
        ('Y', 'Z')
    ], qubit)


def F1(state,
       qubit: int) -> None:
    """
    Square root of Z.

    X -> Y
    Z -> X
    Y -> Z

    Args:
        state (PauliFaultProp):  The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """

    switch(state, [
        ('X', 'Y'),
        ('Z', 'X'),
        ('Y', 'Z')
    ], qubit)


def F2(state,
       qubit: int) -> None:
    """
    Square root of Z.

    X -> -Z
    Z -> Y
    Y -> -X

    Args:
        state (PauliFaultProp):  The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """

    switch(state, [
        ('X', 'Z'),
        ('Z', 'Y'),
        ('Y', 'X')
    ], qubit)


def F3(state,
       qubit: int) -> None:
    """
    Square root of Z.

    X -> Y    gain Z
    Z -> -X   loose Z gain X
    Y -> -Z   loose X

    Args:
        state (PauliFaultProp):  The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """

    switch(state, [
        ('X', 'Y'),
        ('Z', 'X'),
        ('Y', 'Z')
    ], qubit)


def F4(state,
       qubit: int) -> None:
    """
    Square root of Z.

    X -> Z
    Z -> -Y
    Y -> -X

    Args:
        state (PauliFaultProp):  The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """

    switch(state, [
        ('X', 'Z'),
        ('Z', 'Y'),
        ('Y', 'X')
    ], qubit)


def F1d(state,
        qubit: int) -> None:
    """
    Square root of Z.

    X -> Z
    Z -> Y
    Y -> X

    Args:
        state (PauliFaultProp):  The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """

    switch(state, [
        ('X', 'Z'),
        ('Z', 'Y'),
        ('Y', 'X')
    ], qubit)


def F2d(state,
        qubit: int) -> None:
    """
    Square root of Z.

    X -> Z
    Z -> Y
    Y -> X

    Args:
        state (PauliFaultProp):  The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """

    switch(state, [
        ('X', 'Z'),
        ('Z', 'Y'),
        ('Y', 'X')
    ], qubit)


def F3d(state,
        qubit: int) -> None:
    """
    Square root of Z.

    X -> Z
    Z -> Y
    Y -> X

    Args:
        state (PauliFaultProp):  The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """

    switch(state, [
        ('X', 'Z'),
        ('Z', 'Y'),
        ('Y', 'X')
    ], qubit)


def F4d(state,
        qubit: int) -> None:
    """
    Square root of Z.

    X -> Z
    Z -> Y
    Y -> X

    Args:
        state (PauliFaultProp):  The class representing the Pauli fault state.
        qubit (int): An integer indexing the qubit being operated on.

    Returns: None

    """

    switch(state, [
        ('X', 'Z'),
        ('Z', 'Y'),
        ('Y', 'X')
    ], qubit)
