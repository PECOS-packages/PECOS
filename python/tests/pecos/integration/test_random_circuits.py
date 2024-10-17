# Copyright 2024 The PECOS Developers
# Copyright 2018 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract
# DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from __future__ import annotations

import numpy as np
from pecos.simulators import SparseSimPy, SparseSimRs


def test_random_circuits():
    state_sims = []

    # Add wrapped CHP
    try:
        from pecos.state_sims.cychp import State as StateCHP

        state_sims.append(StateCHP)

    except ImportError:
        pass

    # Add wrapped GraphSim
    try:
        from pecos.state_sims.cygraphsim import State as StateGraph

        state_sims.append(StateGraph)

    except ImportError:
        pass

    # Add wrapped C++ version of SparseStabSim
    try:
        from pecos.state_sims.cysparsesim import State as StateCySparse

        state_sims.append(StateCySparse)

    except ImportError:
        pass

    try:
        from pecos.state_sims.cysparsesim_simple import State as StateCySparseSim

        state_sims.append(StateCySparseSim)

    except ImportError:
        pass

    state_sims.append(SparseSimPy)
    state_sims.append(SparseSimRs)

    assert run_circuit_test(state_sims, num_qubits=10, circuit_depth=50)


def run_circuit_test(
    state_sims: list,
    num_qubits: int,
    circuit_depth: int,
    trials: int = 1000,
    gates: list[str] | None = None,
) -> bool:
    if gates is None:
        gates = ["H", "S", "CNOT", "measure Z", "init |0>"]

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
                print("seed=", seed)
                print(circuit)
                return False

    return True


def get_qubits(num_qubits: int, size: int):
    return np.random.choice(list(range(num_qubits)), size, replace=False)


def generate_circuit(
    gates: list,
    num_qubits: int,
    circuit_depth: int,
):
    circuit_elements = list(np.random.choice(gates, circuit_depth))

    circuit = []

    for element in circuit_elements:
        if element == "CNOT":
            q = get_qubits(num_qubits, 2)
        else:
            q = int(get_qubits(num_qubits, 1)[0])

        circuit.append((element, q))

    return circuit


def run_a_circuit(
    num_qubits: int,
    state_rep,
    circuit,
    *,
    verbose: bool = False,
):
    state = state_rep(num_qubits)
    measurements = []

    if isinstance(state, SparseSimRs):
        state.bindings["measure Z"] = state.bindings["MZForced"]
        state.bindings["init |0>"] = state.bindings["PZForced"]

    for element, q in circuit:
        m = -1
        if element == "measure Z":
            m = state.run_gate(element, {q}, forced_outcome=0)
            m = m.get(q, 0)
            measurements.append(m)

        elif element == "init |0>":
            if isinstance(q, np.ndarray):
                q = tuple(q)

            state.run_gate(element, {q}, forced_outcome=0)

        else:
            if isinstance(q, np.ndarray):
                q = tuple(q)

            state.run_gate(element, {q})

        if verbose:
            print("\ngate", element, q, "->")
            if m > -1:
                print("result:", m)

            try:
                state.print_tableau(state.stabs)
                print("..")
                state.print_tableau(state.destabs)
            except AttributeError:
                pass
    if verbose:
        print("\n!!! DONE\n\n")

    return measurements
