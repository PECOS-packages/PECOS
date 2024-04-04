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
from pecos.simulators import BasicSV

# Try to import the requirements for CuStateVec
try:
    import cuquantum

    imported_cuquantum = vparse(cuquantum._version.__version__) >= vparse("23.6.0")  # noqa: SLF001
    import cupy as cp  # noqa: F401

    imported_cupy = vparse(version("cupy")) >= vparse("10.4.0")
    from pecos.simulators import CuStateVec

    custatevec_ready = imported_cuquantum and imported_cupy
except ImportError:
    custatevec_ready = False


def verify(simulator, qc: QuantumCircuit, final_vector: np.ndarray) -> None:
    if simulator == "BasicSV":
        sim = BasicSV(len(qc.qudits))
    elif simulator == "CuStateVec" and custatevec_ready:
        sim = CuStateVec(len(qc.qudits))
    else:
        pytest.skip(f"Requirements to test {simulator} are not met.")

    sim.run_circuit(qc)
    assert np.allclose(sim.vector, final_vector)


def check_measurement(simulator, qc: QuantumCircuit, final_results: dict[int, int] | None = None) -> None:
    if simulator == "BasicSV":
        sim = BasicSV(len(qc.qudits))
    elif simulator == "CuStateVec" and custatevec_ready:
        sim = CuStateVec(len(qc.qudits))
    else:
        pytest.skip(f"Requirements to test {simulator} are not met.")

    results = sim.run_circuit(qc)

    if final_results is not None:
        assert results == final_results

    state = 0
    for q, value in results.items():
        state += value * 2 ** (sim.num_qubits - 1 - q)
    final_vector = np.zeros(shape=(2**sim.num_qubits,))
    final_vector[state] = 1

    abs_values_vector = [abs(x) for x in sim.vector]

    assert np.allclose(abs_values_vector, final_vector)


@pytest.mark.parametrize(
    "simulator",
    [
        "BasicSV",
        "CuStateVec",
    ],
)
def test_init(simulator):
    qc = QuantumCircuit()
    qc.append({"Init": {0, 1, 2, 3}})

    final_vector = np.zeros(shape=(2**4,))
    final_vector[0] = 1

    verify(simulator, qc, final_vector)


@pytest.mark.parametrize(
    "simulator",
    [
        "BasicSV",
        "CuStateVec",
    ],
)
def test_H_measure(simulator):
    qc = QuantumCircuit()
    qc.append({"H": {0, 1, 2, 3, 4}})
    qc.append({"Measure": {0, 1, 2, 3, 4}})

    check_measurement(simulator, qc)


@pytest.mark.parametrize(
    "simulator",
    [
        "BasicSV",
        "CuStateVec",
    ],
)
def test_comp_basis_circ_and_measure(simulator):
    qc = QuantumCircuit()
    qc.append({"Init": {0, 1, 2, 3}})

    # Step 1
    qc.append({"X": {0, 2}})  # |0000> -> |1010>

    final_vector = np.zeros(shape=(2**4,))
    final_vector[10] = 1  # |1010>

    verify(simulator, qc, final_vector)

    # Step 2
    qc.append({"CX": {(2, 1)}})  # |1010> -> |1110>

    final_vector = np.zeros(shape=(2**4,))
    final_vector[14] = 1  # |1110>

    verify(simulator, qc, final_vector)

    # Step 3
    qc.append({"SWAP": {(0, 3)}})  # |1110> -> |0111>

    final_vector = np.zeros(shape=(2**4,))
    final_vector[7] = 1  # |0111>

    verify(simulator, qc, final_vector)

    # Step 4
    qc.append({"CX": {(0, 2)}})  # |0111> -> |0111>

    final_vector = np.zeros(shape=(2**4,))
    final_vector[7] = 1  # |0111>

    verify(simulator, qc, final_vector)

    # Step 5
    qc.append({"Init": {1}})  # |0111> -> |0011>

    final_vector = np.zeros(shape=(2**4,))
    final_vector[3] = 1  # |0011>

    verify(simulator, qc, final_vector)

    # Step 6
    qc.append({"SWAP": {(1, 2)}})  # |0011> -> |0101>

    final_vector = np.zeros(shape=(2**4,))
    final_vector[5] = 1  # |0011>

    verify(simulator, qc, final_vector)

    # Measure
    qc.append({"Measure": {0, 1, 2, 3}})

    final_results = {1: 1, 3: 1}  # |0101>

    check_measurement(simulator, qc, final_results)


@pytest.mark.parametrize(
    "simulator",
    [
        "BasicSV",
        "CuStateVec",
    ],
)
def test_all_gate_circ(simulator):
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

    check_measurement(simulator, qc)
