from ...circuits import QuantumCircuit


def find_logical_signs(state,
                       logical_circuit: QuantumCircuit) -> int:
    """
    Find the sign of the logical operator.

    Args:
        state:
        logical_circuit:

    Returns:

    """
    if len(logical_circuit) != 1:
        raise Exception('Logical operators are expected to only have one tick.')

    logical_xs = set([])
    logical_zs = set([])

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

    anticom = len(state.faults['X'] & logical_zs)
    anticom += len(state.faults['Y'] & logical_zs)
    anticom += len(state.faults['Y'] & logical_xs)
    anticom += len(state.faults['Z'] & logical_xs)

    return anticom % 2
