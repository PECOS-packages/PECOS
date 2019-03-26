from projectq.ops import QubitOperator
from ...circuits import QuantumCircuit


def find_logical_signs(state,
                       logical_circuit: QuantumCircuit,
                       allow_float=False) -> int:
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

    op_string = []

    for symbol, gate_locations, _ in logical_circuit.items():

        if symbol == 'X':
            logical_xs.update(gate_locations)
            for loc in gate_locations:
                op_string.append('X%s' % loc)
        elif symbol == 'Z':
            logical_zs.update(gate_locations)
            for loc in gate_locations:
                op_string.append('Z%s' % loc)
        elif symbol == 'Y':
            logical_xs.update(gate_locations)
            logical_zs.update(gate_locations)
            for loc in gate_locations:
                op_string.append('Y%s' % loc)
        else:
            raise Exception('Can not currently handle logical operator with operator "%s"!' % symbol)

    op_string = ' '.join(op_string)
    state.eng.flush()
    result = state.eng.backend.get_expectation_value(QubitOperator(op_string), state.qureg)

    if not allow_float:
        result = round(result, 4)
        if result == -1:
            return 1
        elif result == 1:
            return 0
        else:
            raise Exception('Unexpected result found!')

    return result
