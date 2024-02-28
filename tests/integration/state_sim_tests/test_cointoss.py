# Copyright 2023 The PECOS Developers
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

from importlib.metadata import version

import numpy as np
import pytest
from packaging.version import parse as vparse
from pecos.circuits import QuantumCircuit
from pecos.simulators import CoinToss


def test_fixed_prob():
    qc = QuantumCircuit()
    qc.append({"Init": {0, 1, 2, 3, 4}})
    qc.append({"Measure": {0, 1, 2, 3, 4}})

    # Probability = 0
    sim = CoinToss(len(qc.qudits), prob=0.0)
    results = sim.run_circuit(qc)
    assert len(results) == 0  # No measurement returned 1

    # Probability = 1
    sim = CoinToss(len(qc.qudits), prob=1.0)
    results = sim.run_circuit(qc)
    assert len(results) == len(qc.qudits)  # All measurements returned 1


def test_all_gate_circ():
    qc = QuantumCircuit()

    # Apply each gate once
    qc.append({"Init": {0, 1, 2, 3, 4}})
    qc.append({"SZZ": {(4, 2)}})
    qc.append({"RX": {0, 2}}, angles=(np.pi / 4,))
    qc.append({"SXXdg": {(0, 3)}})
    qc.append({"RY": {0, 3}}, angles=(np.pi / 8,))
    qc.append({"RZZ": {(0, 3)}}, angles=(np.pi / 16,))
    qc.append({"RZ": {1, 4}}, angles=(np.pi / 16,))
    qc.append({"R1XY": {2}}, angles=(np.pi / 16, np.pi / 2))
    qc.append({"I": {0, 1, 3}})
    qc.append({"X": {1, 2}})
    qc.append({"Y": {3, 4}})
    qc.append({"CY": {(2, 3)}})
    qc.append({"SYY": {(1, 4)}})
    qc.append({"Z": {2, 0}})
    qc.append({"H": {3, 1}})
    qc.append({"RYY": {(2, 1)}}, angles=(np.pi / 8,))
    qc.append({"SZZdg": {(3, 1)}})
    qc.append({"F": {0, 2, 4}})
    qc.append({"CX": {(0, 1)}})
    qc.append({"Fdg": {3, 1}})
    qc.append({"SYYdg": {(1, 3)}})
    qc.append({"SX": {1, 2}})
    qc.append({"R2XXYYZZ": {(0, 4)}}, angles=(np.pi / 4, np.pi / 16, np.pi / 2))
    qc.append({"SY": {3, 4}})
    qc.append({"SZ": {2, 0}})
    qc.append({"SZdg": {1, 2}})
    qc.append({"CZ": {(1, 3)}})
    qc.append({"SXdg": {3, 4}})
    qc.append({"SYdg": {2, 0}})
    qc.append({"T": {0, 2, 4}})
    qc.append({"SXX": {(0, 2)}})
    qc.append({"SWAP": {(4, 0)}})
    qc.append({"Tdg": {3, 1}})
    qc.append({"RXX": {(1, 3)}}, angles=(np.pi / 4,))

    # Measure
    qc.append({"Measure": {0, 1, 2, 3, 4}})

    # Run
    sim = CoinToss(len(qc.qudits))
    sim.run_circuit(qc)
