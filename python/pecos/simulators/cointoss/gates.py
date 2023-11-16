# TODO: Include license information?

from typing import Any

import numpy as np


def ignore_gate(state, qubits: int, **params: Any) -> None:
    """
    Ignore the gate.

    Args:
        state: An instance of ``CoinToss``.
        qubits: The qubits the gate was applied to.

    Returns:

    """


def measure(state, qubits: int, **params: Any):
    """
    Return 0 with probability ``state.prob`` or 1 otherwise.

    Args:
        state: An instance of ``CoinToss``.
        qubit: The qubit the measurement is applied to.

    Returns:

    """
    return 0 if np.random.random() < state.prob else 1
