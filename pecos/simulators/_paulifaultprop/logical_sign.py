from typing import Optional
from ...circuits import QuantumCircuit

def find_logical_signs(state,
                       logical_circuit: QuantumCircuit,
                       delogical_circuit: Optional[QuantumCircuit] = None) -> int:
    """
    Find the sign of the logical operator.

    Args:
        state:
        logical_circuit:
        delogical_circuit:

    Returns:

    """
    if len(logical_circuit) != 1:
        raise Exception('Logical operators are expected to only have one tick.')

    if delogical_circuit and len(delogical_circuit) != 1:
        raise Exception('Delogical operators are expected to only have one tick.')

    logical_xs = set([])
    logical_zs = set([])

    delogical_xs = set([])
    delogical_zs = set([])

    for symbol, gate_locations, _ in logical_circuit.items():

        if symbol == 'X':
            logical_xs.update(gate_locations)
        elif symbol == 'Z':
            logical_zs.update(gate_locations)
        elif symbol == 'Y':
            logical_xs.update(gate_locations)
            logical_zs.update(gate_locations)
        else:
            raise Exception('Can not currently handle logical operator with operator "%s"!' % symbol)

    if delogical_circuit:  # Check the relationship between logical operator and delogical operator.
        for symbol, gate_locations, _ in delogical_circuit.items():

            if symbol == 'X':
                delogical_xs.update(gate_locations)
            elif symbol == 'Z':
                delogical_zs.update(gate_locations)
            elif symbol == 'Y':
                delogical_xs.update(gate_locations)
                delogical_zs.update(gate_locations)
            else:
                raise Exception('Can not currently handle logical operator with operator "%s"!' % symbol)

        # Make sure the logical and delogical anti-commute

        anticom_x = len(logical_xs & delogical_zs) % 2  # Number of common elements modulo 2
        anticom_z = len(logical_zs & delogical_xs) % 2  # Number of common elements modulo 2

        if not ((anticom_x + anticom_z) % 2):
            print('logical Xs: %s logical Zs: %s' % (logical_xs, logical_zs))
            print('delogical Xs: %s delogical Zs: %s' % (delogical_xs, delogical_zs))
            raise Exception("Logical and delogical operators supplied do not anti-commute!")

    anticom = len(state.faults['X'] & logical_zs)
    anticom += len(state.faults['Y'] & logical_zs)
    anticom += len(state.faults['Y'] & logical_xs)
    anticom += len(state.faults['Z'] & logical_xs)

    return anticom % 2
