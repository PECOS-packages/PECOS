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

import numpy as np

from pecos.circuits import QuantumCircuit
from pecos.engines.circuit_runners import TimingRunner
from pecos.simulators import SparseSimPy


def random_circuit_speed(
    state_sim,
    num_qubits,
    circuit_depth,
    trials=10000,
    gates=None,
    seed_start=0,
):
    circuits = generate_circuits(num_qubits, circuit_depth, trials, gates, seed_start)

    times = []
    measurements = []

    circ_sim = TimingRunner()
    for qc in circuits:
        state = SparseSimPy(num_qubits)
        meas = circ_sim.run(state, qc)
        times.append(circ_sim.total_time)
        measurements.append(meas)

    return times, measurements


def generate_circuits(
    num_qubits,
    circuit_depth,
    trials=100000,
    gates=None,
    seed_start=0,
):
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
        np.random.seed(seed)

        circuit_elements = list(np.random.choice(gates, circuit_depth))
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


def get_qubits(num_qubits, size):
    return np.random.choice(list(range(num_qubits)), size, replace=False)
