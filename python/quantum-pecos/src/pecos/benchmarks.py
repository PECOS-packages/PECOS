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

"""Performance benchmarking tools for random quantum circuits.

This module provides utilities for measuring and analyzing the execution speed
and performance characteristics of randomly generated quantum circuits, useful
for benchmarking quantum simulators and circuit execution engines.

Example:
    >>> from pecos.benchmarks import random_circuit_speed, generate_circuits
    >>> from pecos.simulators import SparseSimPy
    >>>
    >>> # Benchmark random circuits
    >>> times, measurements, circuits = random_circuit_speed(
    ...     state_sim=SparseSimPy,
    ...     num_qubits=10,
    ...     circuit_depth=100,
    ...     trials=1000,
    ... )
    >>>
    >>> # Generate circuits for custom benchmarking
    >>> circuits = generate_circuits(
    ...     num_qubits=10,
    ...     circuit_depth=50,
    ...     trials=100,
    ... )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pecos as pc
from pecos.circuits import QuantumCircuit
from pecos.engines.circuit_runners import TimingRunner

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    from pecos.protocols import SimulatorProtocol
    from pecos.typing import Array


def random_circuit_speed(
    state_sim: type[SimulatorProtocol],
    num_qubits: int,
    circuit_depth: int,
    trials: int = 10000,
    gates: Sequence[str] | None = None,
    seed_start: int = 0,
    converter: Callable[[QuantumCircuit], QuantumCircuit] | None = None,
) -> tuple[list[float], list[dict[str, int | list[int]]], list[QuantumCircuit]]:
    """Measure execution speed of random quantum circuits.

    Generates random quantum circuits and measures their execution times
    using a particular simulator for performance benchmarking.

    Args:
        state_sim: Simulator protocol type.
        num_qubits: Number of qubits in the circuits.
        circuit_depth: Number of gates per circuit.
        trials: Number of random circuits to generate and test.
        gates: List of gate types to use, defaults to comprehensive gate set.
        seed_start: Starting seed for random number generation.
        converter: Optional function to convert circuits before execution.

    Returns:
        Tuple of execution times, measurement results, and generated circuits.
    """
    circuits = generate_circuits(num_qubits, circuit_depth, trials, gates, seed_start)

    times = []
    measurements = []

    circ_sim = TimingRunner()
    for qc in circuits:
        circuit_to_run = qc
        if converter is not None:
            circuit_to_run = converter(qc)

        state = state_sim(num_qubits)
        circ_sim.reset_time()
        meas = circ_sim.run(state, circuit_to_run)
        times.append(circ_sim.total_time)
        measurements.append(meas)

    return times, measurements, circuits


def generate_circuits(
    num_qubits: int,
    circuit_depth: int,
    trials: int = 100000,
    gates: Sequence[str] | None = None,
    seed_start: int = 0,
    *,
    iterate: bool = False,
) -> list[QuantumCircuit] | Generator[QuantumCircuit, None, None]:
    """Generate random quantum circuits for performance testing.

    Creates a collection of random quantum circuits with specified parameters,
    using a comprehensive set of quantum gates and operations.

    Args:
        num_qubits: Number of qubits per circuit.
        circuit_depth: Number of gates per circuit.
        trials: Number of circuits to generate.
        gates: Gate types to use, defaults to extensive gate library.
        seed_start: Starting seed for reproducible randomness.
        iterate: Whether to return a generator instead of a list.

    Returns:
        List or generator of randomly generated quantum circuits.
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

            if element in {"CNOT", "CZ", "SWAP", "G", "II", "CY"}:
                q = get_qubits(num_qubits, 2)
                q = (int(q[0]), int(q[1]))

            else:
                q = int(get_qubits(num_qubits, 1))

                if element in {"measure Z", "measure X", "measure Y"}:
                    params = {"gate_kwargs": {"forced_outcome": 0}}

            qc.append(element, {q}, **params)

        if iterate:
            yield qc
        else:
            circuits.append(qc)

    if not iterate:
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


__all__ = [
    "generate_circuits",
    "get_qubits",
    "random_circuit_speed",
]
