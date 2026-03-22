"""Performance benchmarking tools for logical quantum circuits.

This module provides utilities for measuring and analyzing the execution
speed and performance characteristics of logical quantum circuits in
quantum error correction systems.
"""

# Copyright 2019 The PECOS Developers
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

from typing import TYPE_CHECKING

import pecos as pc
from pecos.circuits import QuantumCircuit
from pecos.engines.circuit_runners import TimingRunner

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pecos.protocols import SimulatorProtocol
    from pecos.typing import Array


def random_circuit_speed(
    state_sim: type[SimulatorProtocol],
    num_qubits: int,
    circuit_depth: int,
    trials: int = 10000,
    gates: Sequence[str] | None = None,
    seed_start: int = 0,
) -> tuple[list[float], list[dict[str, int | list[int]]]]:
    """Measure execution speed of random quantum circuits.

    Generates random quantum circuits and measures their execution times
    using particular simulator for performance benchmarking.

    Args:
        state_sim: Simulator protocol type (unused in current implementation).
        num_qubits: Number of qubits in the circuits.
        circuit_depth: Number of gates per circuit.
        trials: Number of random circuits to generate and test.
        gates: List of gate types to use, defaults to comprehensive gate set.
        seed_start: Starting seed for random number generation.

    Returns:
        Tuple of execution times and measurement results for each circuit.
    """
    circuits = generate_circuits(num_qubits, circuit_depth, trials, gates, seed_start)

    times = []
    measurements = []

    circ_sim = TimingRunner()
    for qc in circuits:
        state = state_sim(num_qubits)
        meas = circ_sim.run(state, qc)
        times.append(circ_sim.total_time)
        measurements.append(meas)

    return times, measurements


def generate_circuits(
    num_qubits: int,
    circuit_depth: int,
    trials: int = 100000,
    gates: Sequence[str] | None = None,
    seed_start: int = 0,
) -> list[QuantumCircuit]:
    """Generate random quantum circuits for performance testing.

    Creates a collection of random quantum circuits with specified parameters,
    using a comprehensive set of quantum gates and operations.

    Args:
        num_qubits: Number of qubits per circuit.
        circuit_depth: Number of gates per circuit.
        trials: Number of circuits to generate.
        gates: Gate types to use, defaults to extensive gate library.
        seed_start: Starting seed for reproducible randomness.

    Returns:
        List of randomly generated quantum circuits.
    """
    if gates is None:
        gates = [
            "I",
            "X",
            "Y",
            "Z",
            "S",
            "Sd",
            "Q",
            "Qd",
            "R",
            "Rd",
            "H",
            "H1",
            "H2",
            "H3",
            "H4",
            "H5",
            "H6",
            "H+z+x",
            "H-z-x",
            "H+y-z",
            "H-y-z",
            "H-x+y",
            "H-x-y",
            "F1",
            "F1d",
            "F2",
            "F2d",
            "F3",
            "F3d",
            "F4",
            "F4d",
            "CNOT",
            "CZ",
            "SWAP",
            "G",
            "II",
            "measure X",
            "measure Y",
            "measure Z",
            "init |+>",
            "init |->",
            "init |+i>",
            "init |-i>",
            "init |0>",
            "init |1>",
        ]

    circuits = []

    for seed in range(seed_start, seed_start + trials):
        pc.random.seed(seed)

        circuit_elements = list(pc.random.choice(gates, circuit_depth))
        qc = QuantumCircuit()

        for element in circuit_elements:
            params = {}

            if element in {"CNOT", "CZ", "SWAP", "G", "II"}:
                q = get_qubits(num_qubits, 2)
                q = tuple(q)

            else:
                q = int(get_qubits(num_qubits, 1))

                if element in {"measure Z", "measure X", "measure Y"}:
                    params = {"gate_kwargs": {"forced_outcome": 0}}

            qc.append(element, {q}, **params)

        circuits.append(qc)

    return circuits


def get_qubits(num_qubits: int, size: int) -> Array:
    """Get random qubit indices without replacement.

    Args:
        num_qubits: Total number of qubits available.
        size: Number of qubits to select.

    Returns:
        Array of randomly selected qubit indices.
    """
    return pc.random.choice(list(range(num_qubits)), size, replace=False)
