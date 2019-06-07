from typing import Any, Union, Tuple


def meas_x(state,
           qubit: int,
           **params: Any) -> int:
    """
    Measurement in the X basis.

    Args:
        state:
        qubit:
        **params:

    Returns:

    """

    if qubit in state.faults['Z'] or qubit in state.faults['Y']:
        return 1
    else:
        return 0


def meas_z(state,
           qubit: int,
           **params: Any) -> int:
    """
    Measurement in the Z basis.

    Args:
        state:
        qubit:
        **params:

    Returns:

    """

    if qubit in state.faults['X'] or qubit in state.faults['Y']:
        return 1
    else:
        return 0


def meas_y(state,
           qubit: int,
           **params: Any):
    """
    Measurement in the Y basis.

    Args:
        state:
        qubit:
        **params:

    Returns:

    """
    if qubit in state.faults['X'] or qubit in state.faults['Z']:
        return 1
    else:
        return 0


def meas_pauli(state,
               qubits: Union[int, Tuple[int, ...]],
               **params: Any) -> int:

    pauli = params['Pauli']

    if isinstance(qubits, int) and pauli not in ['X', 'Y', 'Z']:
        raise Exception('Pauli for a single qubit measurement must be \'X\', \'Y\' or \'Z\'!')

    if pauli in ['X', 'Y', 'Z']:
        pauli = pauli * len(qubits)
    else:
        if len(pauli) == len(qubits) + 1:
            # last qubit is considered the syndrome ancilla
            qubits = qubits[:-1]
        elif len(pauli) != len(qubits):
            raise Exception('The Pauli operator needs to be the size of the qubits it is acting on or a single type.')

    meas = 0

    for q, p in zip(qubits, pauli):
        if p == 'X':
            meas += meas_x(state, q)
        elif p == 'Z':
            meas += meas_z(state, q)
        elif p == 'Y':
            meas += meas_y(state, q)
        else:
            raise Exception('Pauli symbol not supported!')

    return meas % 2


def force_output(state,
                 qubit: int,
                 forced_output: int = -1) -> int:
    """
    Outputs value.

    Used for error generators to generate outputs when replacing measurements.

    Args:
        state:
        qubit:
        forced_output:

    Returns:

    """
    return forced_output
