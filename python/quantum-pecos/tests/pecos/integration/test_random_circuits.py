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

"""Integration tests for random quantum circuit simulations."""
from __future__ import annotations

from typing import Any

import pecos as pc
from pecos.simulators import SparseSim, SparseSimCpp, SparseSimPy


def test_random_circuits() -> None:
    """Test random quantum circuits on different simulators."""
    state_sims: list[type[Any]] = []

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
    state_sims.append(SparseSim)
    state_sims.append(SparseSimCpp)

    assert run_circuit_test(state_sims, num_qubits=10, circuit_depth=50)


def run_circuit_test(
    state_sims: list[type[Any]],
    num_qubits: int,
    circuit_depth: int,
    trials: int = 1000,
    gates: list[str] | None = None,
) -> bool:
    """Run circuit test comparing different simulators."""
    if gates is None:
        gates = ["H", "S", "CNOT", "measure Z", "init |0>"]

    for seed in range(trials):
        pc.random.seed(seed)
        circuit = generate_circuit(gates, num_qubits, circuit_depth)

        measurements = []
        for _i, state_sim in enumerate(state_sims):
            pc.random.seed(seed)
            verbose = False  # Can set to True for debugging
            meas = run_a_circuit(
                num_qubits,
                state_sim,
                circuit,
                _test_seed=seed,
                verbose=verbose,
            )
            measurements.append(meas)

        meas0 = measurements[0]
        for _i, meas in enumerate(measurements[1:], 1):
            if meas0 != meas:
                return False

    return True


def get_qubits(num_qubits: int, size: int) -> pc.Array:
    """Get random qubit indices for gate operations."""
    return pc.random.choice(list(range(num_qubits)), size, replace=False)


def generate_circuit(
    gates: list[str],
    num_qubits: int,
    circuit_depth: int,
) -> list[tuple[str, int | pc.Array]]:
    """Generate a random quantum circuit with specified gates and depth."""
    circuit_elements = list(pc.random.choice(gates, circuit_depth))

    circuit = []

    for element in circuit_elements:
        q = (
            get_qubits(num_qubits, 2)
            if element == "CNOT"
            else int(get_qubits(num_qubits, 1)[0])
        )

        circuit.append((element, q))

    return circuit


def run_a_circuit(
    num_qubits: int,
    state_rep: type[Any],
    circuit: list[tuple[str, int | pc.Array]],
    *,
    verbose: bool = False,
    _test_seed: int | None = None,  # Unused - kept for API compatibility
) -> list[int]:
    """Run a quantum circuit on a specific simulator and return measurements."""
    state = state_rep(num_qubits)
    measurements = []

    if isinstance(state, SparseSim | SparseSimCpp):
        state.bindings["measure Z"] = state.bindings["MZForced"]
        state.bindings["init |0>"] = state.bindings.get(
            "PZForced",
            state.bindings.get("init |0>"),
        )
        # Don't set seed for C++ simulator - use numpy random for forced outcomes instead
        # if isinstance(state, SparseSimCpp) and hasattr(state, 'set_seed') and test_seed is not None:
        #     # Use the test seed directly for C++ RNG
        #     state.set_seed(test_seed)

    for i, (element, q) in enumerate(circuit):
        m = -1
        if element == "measure Z":
            if (
                verbose and isinstance(state, SparseSimCpp) and i == 26
            ):  # Debug the 27th operation
                pass
                # print(f"\n[DEBUG] Op {i}: {element} on qubit {q}, forcing outcome to 0")
            m = state.run_gate(element, {q}, forced_outcome=0)
            m = m.get(q, 0)
            if verbose and isinstance(state, SparseSimCpp) and i == 26:
                pass
                # print(f"[DEBUG] Result: {m}\n")
            measurements.append(m)

        elif element == "init |0>":
            q_tuple = tuple(q) if isinstance(q, (pc.Array, list)) else q

            state.run_gate(element, {q_tuple}, forced_outcome=0)

        else:
            q_tuple = tuple(q) if isinstance(q, (pc.Array, list)) else q

            state.run_gate(element, {q_tuple})

        if verbose:
            # print("\ngate", element, q, "->")
            if m > -1:
                pass
                # print("result:", m)

            try:
                state.print_tableau(state.stabs)
                # print("..")
                state.print_tableau(state.destabs)
            except AttributeError:
                pass
    if verbose:
        pass
        # print("\n!!! DONE\n\n")

    return measurements
