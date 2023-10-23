# TODO: Include license information?

import random


def ignore_gate(state, qubits):
    """
    Ignore the gate.

    Args:
        state: An instance of ``CoinToss``.
        qubits: The qubits the gate was applied to.

    Returns:

    """
    pass


def measure(state, qubit):
    """
    Return 0 with probability ``state.prob`` or 1 otherwise.

    Args:
        state: An instance of ``CoinToss``.
        qubit: The qubit the measurement is applied to.

    Returns:

    """
    return 0 if random.random() < state.prob else 1


def force_output(state, qubit, forced_output=-1):
    """
    Outputs value.

    Used for error generators to generate outputs when replacing measurements.

    Args:
        state: An instance of ``CoinToss``.
        qubit: The qubit the measurement is applied to.
        forced_output: The desired output of the measurement.

    Returns:

    """
    return forced_output
