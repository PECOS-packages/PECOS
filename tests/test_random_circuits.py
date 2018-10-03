#   Copyright 2018 National Technology & Engineering Solutions of Sandia,
#   LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS,
#   the U.S. Government retains certain rights in this software.
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

import numpy as np
from pecos.simulators import SparseSim as state_sparse


def test_random_circuits():

    state_sims = []

    # Add wrapped CHP
    try:
        from PECOS.state_sims.cychp import State as state_chp
        state_sims.append(state_chp)

    except ImportError:
        pass

    # Add wrapped GraphSim
    try:
        from PECOS.state_sims.cygraphsim import State as state_graph
        state_sims.append(state_graph)

    except ImportError:
        pass

    # Add wrapped C++ version of SparseStabSim
    try:
        from PECOS.state_sims.cysparsesim import State as state_csparse
        state_sims.append(state_csparse)

    except ImportError:
        pass

    try:
        from PECOS.state_sims.cysparsesim_simple import State as state_csparse_simp
        state_sims.append(state_csparse_simp)

    except ImportError:
        pass

    state_sims.append(state_sparse)

    assert run_circuit_test(state_sims, num_qubits=10, circuit_depth=50)


def run_circuit_test(state_sims, num_qubits, circuit_depth, trials=1000, gates=None):

    if gates is None:
        gates = ['H', 'S', 'CNOT', 'measure Z', 'init |0>']

    for seed in range(trials):

        np.random.seed(seed)
        circuit = generate_circuit(gates, num_qubits, circuit_depth)

        measurements = []
        for state_sim in state_sims:
            np.random.seed(seed)
            meas = run_a_circuit(num_qubits, state_sim, circuit)

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
            q = int(get_qubits(num_qubits, 1))

        circuit.append((element, q))

    return circuit


def run_a_circuit(num_qubits, state_rep, circuit, verbose=False):
    state = state_rep(num_qubits)
    measurements = []
    gate_dict = state.gate_dict

    for element, q in circuit:

        m = -1
        if element == 'measure Z':
            m = gate_dict[element](state, q, random_outcome=0)
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
