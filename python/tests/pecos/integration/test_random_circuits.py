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

import numpy as np
from pecos.simulators import CppSparseSimRs, SparseSimPy, SparseSimRs


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
    state_sims.append(SparseSimRs)
    state_sims.append(CppSparseSimRs)

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
        np.random.seed(seed)
        circuit = generate_circuit(gates, num_qubits, circuit_depth)

        measurements = []
        for i, state_sim in enumerate(state_sims):
            np.random.seed(seed)
            verbose = (
                seed == 32 and state_sim.__name__ == "CppSparseSimRs"
            )  # Debug failing case
            meas = run_a_circuit(
                num_qubits,
                state_sim,
                circuit,
                _test_seed=seed,
                verbose=verbose,
            )
            if seed == 32:
                print(
                    f"Simulator {i} ({state_sim.__name__}): {meas[:20]}...",
                )  # Show first 20 measurements
            measurements.append(meas)

        meas0 = measurements[0]
        for i, meas in enumerate(measurements[1:], 1):
            if meas0 != meas:
                print("seed=", seed)
                print("Simulator 0 measurements:", meas0)
                print(f"Simulator {i} measurements:", meas)
                print(f"Simulator types: {[type(s).__name__ for s in state_sims]}")
                print(circuit)
                return False

    return True


def get_qubits(num_qubits: int, size: int) -> np.ndarray:
    """Get random qubit indices for gate operations."""
    return np.random.choice(list(range(num_qubits)), size, replace=False)


def generate_circuit(
    gates: list[str],
    num_qubits: int,
    circuit_depth: int,
) -> list[tuple[str, int | np.ndarray]]:
    """Generate a random quantum circuit with specified gates and depth."""
    circuit_elements = list(np.random.choice(gates, circuit_depth))

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
    circuit: list[tuple[str, int | np.ndarray]],
    *,
    verbose: bool = False,
    _test_seed: int | None = None,  # Unused - kept for API compatibility
) -> list[int]:
    """Run a quantum circuit on a specific simulator and return measurements."""
    state = state_rep(num_qubits)
    measurements = []

    if isinstance(state, SparseSimRs | CppSparseSimRs):
        state.bindings["measure Z"] = state.bindings["MZForced"]
        state.bindings["init |0>"] = state.bindings.get(
            "PZForced",
            state.bindings.get("init |0>"),
        )
        # Don't set seed for C++ simulator - use numpy random for forced outcomes instead
        # if isinstance(state, CppSparseSimRs) and hasattr(state, 'set_seed') and test_seed is not None:
        #     # Use the test seed directly for C++ RNG
        #     state.set_seed(test_seed)

    for i, (element, q) in enumerate(circuit):
        m = -1
        if element == "measure Z":
            if (
                verbose and isinstance(state, CppSparseSimRs) and i == 26
            ):  # Debug the 27th operation
                print(f"\n[DEBUG] Op {i}: {element} on qubit {q}, forcing outcome to 0")
            m = state.run_gate(element, {q}, forced_outcome=0)
            m = m.get(q, 0)
            if verbose and isinstance(state, CppSparseSimRs) and i == 26:
                print(f"[DEBUG] Result: {m}\n")
            measurements.append(m)

        elif element == "init |0>":
            if isinstance(q, np.ndarray):
                q = tuple(q)  # noqa: PLW2901 - convert array to tuple

            state.run_gate(element, {q}, forced_outcome=0)

        else:
            if isinstance(q, np.ndarray):
                q = tuple(q)  # noqa: PLW2901 - convert array to tuple

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
