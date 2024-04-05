import numpy as np
from pecos.simulators import CuStateVec

def run_circuit_test(state_sims, num_qubits, circuit_depth, trials=1000, gates=None):

    if gates is None:
        gates = ['H', 'S', 'CNOT', 'measure Z', 'init |0>']

    for seed in range(trials):

        np.random.seed(seed)
        circuit = generate_circuit(gates, num_qubits, circuit_depth)

        measurements = []
        for state_sim in state_sims:
            np.random.seed(seed)
            meas = run_a_circuit(num_qubits, state_sim, circuit, verbose=True)

            measurements.append(meas)

        meas0 = measurements[0]
        for meas in measurements[1:]:
            if meas0 != meas:
                print('seed=', seed)
                print(circuit)
                return False

    return True


def get_qubits(num_qubits, size):
    return np.random.choice(list(range(num_qubits)), size, replace=False)


def generate_circuit(gates, num_qubits, circuit_depth):
    circuit_elements = list(np.random.choice(gates, circuit_depth))

    circuit = []

    for element in circuit_elements:

        if element == 'CNOT':
            q = get_qubits(num_qubits, 2)
        else:
            q = get_qubits(num_qubits, 1)[0]

        circuit.append((element, q))

    return circuit


def run_a_circuit(num_qubits, state_rep, circuit, verbose=False):
    state = state_rep(num_qubits)
    measurements = []
    gate_dict = state.bindings

    for element, q in circuit:

        m = -1
        if element == 'measure Z':
            m = gate_dict[element](state, q, forced_outcome=0)
            measurements.append(m)

        else:

            if isinstance(q, np.ndarray):
                q = tuple(q)

            gate_dict[element](state, q)

        if verbose:
            print('\ngate', element, q, '->')
            if m > -1:
                print('result:', m)

            try:
                state.print_tableau(state.stabs)
                print('..')
                state.print_tableau(state.destabs)
            except AttributeError:
                pass
    if verbose:
        print('\n!!! DONE\n\n')

    return measurements

if __name__ == '__main__':
    assert run_circuit_test([CuStateVec], num_qubits=10, circuit_depth=50)